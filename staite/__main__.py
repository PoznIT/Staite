"""StAIte CLI — entry point.

Usage:
    python -m staite --config staite.yaml
    python -m staite --config staite.yaml --root /path/to/project
    python -m staite --config staite.yaml --log-level DEBUG
"""

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from staite import __version__
from staite.assembler import assemble, write
from staite.cache import DescriptionCache
from staite.config import load_config
from staite.describer import _CONCURRENCY_ANTHROPIC, _CONCURRENCY_AZURE, DescribeResult, describe_files
from staite.diagram.generator import generate as generate_diagram
from staite.providers import create_provider
from staite.scanner import scan
from staite.synthesizer import SynthesisCache, synthesize

app = typer.Typer(
    name="staite",
    help="Generate a dense AI-readable project state snapshot.",
    add_completion=False,
)

console = Console(stderr=True)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, markup=True)],
    )


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"staite {__version__}")
        raise typer.Exit()


@app.command()
def run(
    config_path: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to staite.yaml config file."),
    ] = Path("staite.yaml"),
    root: Annotated[
        Path | None,
        typer.Option("--root", "-r", help="Project root to scan. Defaults to config file's directory."),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", "-l", help="Logging level (DEBUG, INFO, WARNING, ERROR)."),
    ] = "INFO",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Shorthand for --log-level DEBUG."),
    ] = False,
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = None,
) -> None:
    """Generate a STATE.json snapshot of the project for use as Claude chat context."""
    effective_level = "DEBUG" if verbose else log_level
    _setup_logging(effective_level)
    logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------ config
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    project_root = root or config_path.parent
    logger.debug("Project root: %s", project_root)

    # ------------------------------------------------------------------ scan
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Scanning filesystem…", total=None)
        try:
            file_tree = scan(project_root, config.include, config.exclude)
        except NotADirectoryError as exc:
            console.print(f"[bold red]Scan error:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc

    console.print(f"[green]✓[/green] Scanned {len(file_tree.files)} file(s)")

    if not file_tree.files:
        console.print("[yellow]Warning:[/yellow] No files matched — output will be empty.")

    # ------------------------------------------------------------------ describe
    desc_cache = DescriptionCache.load(project_root / config.cache)

    try:
        provider = create_provider(
            provider_type=config.provider,
            model=config.model,
            azure_endpoint=config.azure.endpoint if config.azure else None,
            azure_api_key=config.azure.api_key if config.azure else None,
        )
    except (ValueError, ImportError) as exc:
        console.print(f"[bold red]Provider error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    async def _run_async() -> tuple[DescribeResult, str, str]:
        async with provider:
            describe_result = await describe_files(
                provider=provider,
                root=project_root,
                rel_paths=file_tree.files,
                cache=desc_cache,
                concurrency=_CONCURRENCY_AZURE if config.provider == "azure" else _CONCURRENCY_ANTHROPIC,
            )

            synthesis_cache = SynthesisCache.load(project_root / config.synthesis_cache)
            synthesis_result = await synthesize(
                provider=provider,
                project_name=config.project_name,
                instructions=config.instructions,
                user_conventions=config.conventions,
                tree_lines=file_tree.tree_lines,
                descriptions=describe_result.descriptions,
                cache=synthesis_cache,
                miss_count=describe_result.cache_miss_count,
                regen_threshold=config.regen_threshold,
            )
            synthesis_cache.save()

        return describe_result, synthesis_result.use_cases, synthesis_result.conventions_ai

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"Describing {len(file_tree.files)} file(s) + synthesising…", total=None)
        try:
            describe_result, use_cases, ai_conventions = asyncio.run(_run_async())
        except Exception as exc:
            console.print(f"[bold red]API error:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc

    console.print(
        f"[green]✓[/green] Descriptions ready "
        f"({describe_result.cache_miss_count} new, {desc_cache.size} cached total)"
    )
    console.print(f"[dim]Token usage: {provider.usage}[/dim]")
    desc_cache.save()

    # ------------------------------------------------------------------ diagram
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Generating diagram…", total=None)
        diagram = generate_diagram(file_tree.files, project_root)

    console.print("[green]✓[/green] Diagram generated")

    # ------------------------------------------------------------------ assemble
    state_json = assemble(
        project_name=config.project_name,
        instructions=config.instructions,
        user_conventions=config.conventions,
        use_cases=use_cases,
        ai_conventions=ai_conventions,
        tree_lines=file_tree.tree_lines,
        diagram=diagram,
        descriptions=describe_result.descriptions,
    )

    output_path = project_root / config.output
    write(state_json, output_path)

    console.print(f"[bold green]✓ State written to {output_path}[/bold green]")


@app.command()
def serve(
    state: Annotated[
        list[Path],
        typer.Option(
            "--state",
            help="Path to a STATE.json file to index. Repeat to serve multiple projects.",
        ),
    ] = [],
    db: Annotated[
        Path,
        typer.Option("--db", help="Central ChromaDB directory shared across all projects."),
    ] = Path.home() / ".staite" / "vector_db",
    transport: Annotated[
        str,
        typer.Option("--transport", "-t", help="MCP transport: 'stdio', 'sse', or 'http'."),
    ] = "stdio",
    host: Annotated[
        str,
        typer.Option("--host", help="Bind host for HTTP/SSE transport."),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Bind port for HTTP/SSE transport."),
    ] = 8080,
    log_level: Annotated[
        str,
        typer.Option("--log-level", "-l", help="Logging level (DEBUG, INFO, WARNING, ERROR)."),
    ] = "INFO",
) -> None:
    """Start the StAIte MCP server for one or more projects.

    Builds vector indexes from STATE.json files on first run, then serves four
    tools — search, get_file, get_overview, list_projects — via the chosen transport.

    Examples:

      # Single project (stdio for Claude Code CLI):
      staite serve --state .staite/STATE.json

      # Multiple projects sharing a central index:
      staite serve \\
        --state ~/Projects/api/.staite/STATE.json \\
        --state ~/Projects/frontend/.staite/STATE.json \\
        --db ~/.staite/vector_db

      # HTTP transport for Claude.ai / remote clients:
      staite serve --state .staite/STATE.json --transport http --port 8080
    """
    _setup_logging(log_level)

    if not state:
        console.print("[bold red]Error:[/bold red] provide at least one --state path.")
        raise typer.Exit(code=1)

    if transport not in ("stdio", "sse", "http"):
        console.print(
            f"[bold red]Error:[/bold red] --transport must be 'stdio', 'sse', or 'http', got {transport!r}"
        )
        raise typer.Exit(code=1)

    try:
        from staite.mcp_server import serve as _serve
    except ImportError as exc:
        console.print(f"[bold red]Import error:[/bold red] {exc}")
        console.print("Install vector deps: [bold]pip install 'staite[vector]'[/bold]")
        raise typer.Exit(code=1) from exc

    if transport != "stdio":
        console.print(f"[green]StAIte MCP server listening on http://{host}:{port}[/green]")

    try:
        _serve(state_paths=list(state), db_path=db, transport=transport, host=host, port=port)
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[bold red]Server error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
