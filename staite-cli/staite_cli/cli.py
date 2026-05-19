"""StAIte CLI — entry point.

Usage:
    python -m staite --config staite.yml
    python -m staite --config staite.yml --root /path/to/project
    python -m staite --config staite.yml --log-level DEBUG
"""

import asyncio
import logging
from urllib import request
from pathlib import Path
from typing import Annotated
from urllib.error import HTTPError

import typer
import json
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from staite_cli.assembler import assemble, write
from staite_cli.cache import DescriptionCache
from staite_cli.describer import CONCURRENCY_ANTHROPIC, CONCURRENCY_AZURE, DescribeResult, describe_files
from staite_cli.diagram.generator import generate as generate_diagram
from staite_cli.providers import create_provider
from staite_cli.scanner import scan
from staite_cli.synthesizer import SynthesisCache, synthesize
from staite_cli.config.run_config import RunConfig

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


@app.command()
def run(
    config_path: Annotated[
      Path,
      typer.Option("--config", "-c", help="Path to staite.yml config file."),
    ] = Path("staite.yml"),
    root: Annotated[
      Path | None,
      typer.Option("--root", "-r", help="Project root to scan. Defaults to config file's directory."),
    ] = None,
    regen_synthesis: Annotated[
      bool,
      typer.Option("--regen_synthesis", "-s", help="Force regeneration of the synthesis")
    ] = False,
    log_level: Annotated[
      str,
      typer.Option("--log-level", "-l", help="Logging level (DEBUG, INFO, WARNING, ERROR)."),
    ] = "INFO",
    verbose: Annotated[
      bool,
      typer.Option("--verbose", "-v", help="Shorthand for --log-level DEBUG."),
    ] = False,
) -> None:
  """Generate a STATE.json snapshot of the project for use as Claude chat context."""
  effective_level = "DEBUG" if verbose else log_level
  _setup_logging(effective_level)
  logger = logging.getLogger(__name__)

  # ------------------------------------------------------------------ config
  try:
    config = RunConfig()
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
      anthropic= config.anthropic,
      azure = config.azure,
      ollama = config.ollama
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
        concurrency=CONCURRENCY_AZURE if config.provider == "azure" else CONCURRENCY_ANTHROPIC,
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
        force_regen=regen_synthesis
      )
      synthesis_cache.save()

    return describe_result, synthesis_result.use_cases_diagram, synthesis_result.conventions_ai

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
def update(
    state: Annotated[
      Path,
      typer.Argument(help="Path to the STATE.json file to push."),
    ],
    url: Annotated[
      str,
      typer.Option("--url", "-u", help="URL of the running MCP server's /update endpoint."),
    ] = "http://localhost:8080/update",
    log_level: Annotated[
      str,
      typer.Option("--log-level", "-l", help="Logging level (DEBUG, INFO, WARNING, ERROR)."),
    ] = "INFO",
) -> None:
  """Push a STATE.json to a running MCP server to re-index without restarting.

  The server must be running with --transport sse or --transport http.

  Examples:

    staite update .staite/STATE.json
    staite update .staite/STATE.json --url http://myserver:8080/update
  """
  _setup_logging(log_level)

  if not state.exists():
    console.print(f"[bold red]Error:[/bold red] STATE.json not found at {state}")
    raise typer.Exit(code=1)

  try:
    payload = state.read_bytes()
    req = request.Request(
      url,
      data=payload,
      headers={"Content-Type": "application/json"},
      method="POST",
    )
    with request.urlopen(req, timeout=30) as resp:
      body = json.loads(resp.read())
  except HTTPError as exc:
    body = json.loads(exc.read())
    console.print(f"[bold red]Server error {exc.code}:[/bold red] {body.get('message', exc)}")
    raise typer.Exit(code=1) from exc
  except OSError as exc:
    console.print(f"[bold red]Connection error:[/bold red] {exc}")
    console.print(f"Is the server running at [bold]{url}[/bold]?")
    raise typer.Exit(code=1) from exc

  console.print(
    f"[bold green]✓ Updated project [cyan]{body['project']}[/cyan]"
    f" — {body['chunks']} chunks indexed[/bold green]"
  )


@app.command()
def conf():
  config = RunConfig()


if __name__ == "__main__":
  app()
