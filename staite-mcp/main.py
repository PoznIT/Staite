import argparse
import logging
import sys
import tomllib
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from src.mcp_server import mcp_server
from src.config import AppSettings

console = Console(stderr=True)

DEFAULT_CONFIG_PATH = Path(".staite/config.toml")


def _setup_logging(level: str) -> None:
  logging.basicConfig(
    level=level.upper(),
    format="%(message)s",
    handlers=[RichHandler(console=console, show_path=False, markup=True)],
  )


def _load_config(config_path: Path) -> dict:
  if not config_path.exists():
    console.print(f"[bold red]Error:[/bold red] config file not found: {config_path}")
    sys.exit(1)

  with config_path.open("rb") as f:
    return tomllib.load(f)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--config", "-c",
    type=Path,
    default=Path("bundle/config.yml"),
    help="Path to YAML config file",
  )
  args = parser.parse_args()
  config = AppSettings.from_config_file(args.config)

  _setup_logging(config.log_level)

  if config.server.transport not in ("sse", "http"):
    console.print(
      f"[bold red]Error:[/bold red] transport must be 'sse', or 'http', got {config.server.transport!r}"
    )
    sys.exit(1)

  hint = " (push state with 'staite update')"
  path = "/mcp" if config.server.transport == "http" else "/sse"
  console.print(f"[green]StAIte MCP server listening on http://{config.server.host}:{config.server.port}{path}[/green]{hint}")

  try:
    mcp_server(config)
  except (FileNotFoundError, RuntimeError) as exc:
    console.print(f"[bold red]Server error:[/bold red] {exc}")
    sys.exit(1)


if __name__ == "__main__":
  main()
