# StAIte

Generate a dense, AI-readable snapshot of your project and use it as context in Claude chat — without burning tokens re-explaining your codebase every conversation.

## The problem

Claude Code is powerful but expensive in tokens. Claude chat is cheap but stateless — every new conversation starts from zero, and maintaining context manually is tedious.

StAIte solves this by scanning your project and producing a `STATE.json` file you paste into your Claude project instructions. The AI then knows your project structure, what every file does, how components connect, and what conventions you follow — before you type a single word.

The optional MCP server takes this further: semantic search over file descriptions lets Claude (or any MCP client) query your codebase directly, and supports **multiple projects simultaneously** so you can ask architecture-level questions across your entire stack.

## How it works

1. **Scan** the filesystem (no AI, just pathspec pattern matching)
2. **Describe** each file with 2–6 sentences via a cheap AI model — skipping files unchanged since the last run (SHA-256 hash cache)
3. **Diagram** the directory tree and intra-project dependencies as Mermaid
4. **Synthesise** use-case scenarios and coding conventions (regenerated only when enough files changed)
5. **Assemble** everything into `STATE.json`
6. *(Optional)* **Serve** the index as an MCP server for semantic search

---

## Installation

Requires Python 3.11+.

```bash
git clone <this repo>
cd StAIte
pip install -e .
```

For the MCP server and vector search:

```bash
pip install -e ".[vector]"
```

For Azure AI Foundry support:

```bash
pip install -e ".[azure]"
```

---

## Quick start

**1. Copy and edit the example config:**

```bash
cp staite.example.yaml staite.yaml
```

**2. Set your API key:**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or add it to a .env file in the same directory as staite.yaml
```

**3. Generate the snapshot:**

```bash
staite run --config staite.yaml
```

The state file is written to `.staite/STATE.json` (configurable). Paste its contents into your Claude project instructions.

---

## MCP server

The MCP server embeds `STATE.json` into a ChromaDB vector index and exposes four tools to any MCP client (Claude Code, Claude.ai, etc.):

| Tool | Description |
|------|-------------|
| `search(query, k=5, project=None)` | Semantic search over file descriptions. Omit `project` to search across all loaded projects. |
| `get_file(path, project=None)` | Return the AI description for a specific file path. |
| `get_overview(project)` | Return use-cases, conventions, and architecture diagram for a project. |
| `list_projects()` | List all indexed projects and their chunk counts. |

### Single project

```bash
staite serve --state .staite/STATE.json
```

### Multiple projects

Point the server at multiple `STATE.json` files — each gets its own collection in a shared central index at `~/.staite/vector_db`:

```bash
staite serve \
  --state ~/Projects/api/.staite/STATE.json \
  --state ~/Projects/frontend/.staite/STATE.json \
  --state ~/Projects/infra/.staite/STATE.json
```

### HTTP transport (for Claude.ai or remote clients)

```bash
staite serve --state .staite/STATE.json --transport http --port 8080
```

### Docker

```bash
docker compose up
```

The `compose.yaml` mounts `.staite/STATE.json` and starts the server on port 8080.

---

## Configuration

All options live in `staite.yaml`.

```yaml
project_name: MyProject

# Files to include — gitignore-style glob patterns
include:
  - src/**
  - lib/**
  - "*.py"

# Files to exclude (applied after include)
exclude:
  - "**/__pycache__/**"
  - "**/*.pyc"
  - node_modules/**
  - dist/**

# Free-form context for the AI — architecture decisions, entry points,
# things that aren't obvious from the code
instructions: |
  FastAPI backend + React frontend.
  Entry point: src/backend/main.py.
  Do NOT suggest changes to src/generated/ — auto-generated files.

# User-defined coding conventions, merged with AI-inferred ones in the output
conventions: |
  All DB calls go through the repository layer in src/backend/repositories/.
  Errors are always raised as AppError, never raw exceptions.

# Provider: "anthropic" (default) or "azure"
provider: anthropic

# Model for descriptions and synthesis. Haiku is recommended — cheap and fast.
model: claude-haiku-4-5-20251001

# Regenerate use-cases and conventions only if this fraction of files changed.
# 0.2 = regenerate when >20% of files were modified since last run.
regen_threshold: 0.2

output: .staite/STATE.json
cache: .staite/cache.json
synthesis_cache: .staite/synthesis.json
```

---

## Providers

### Anthropic (default)

```yaml
provider: anthropic
```

Requires `ANTHROPIC_API_KEY` — set as an environment variable or in a `.env` file next to `staite.yaml`.

### Azure AI Foundry

```bash
pip install -e ".[azure]"
```

```yaml
provider: azure
azure:
  endpoint: https://<resource>.services.ai.azure.com/models
  api_key: sk-...   # optional — see auth order below
model: claude-3-5-haiku
```

**Authentication order** (first match wins):

| Priority | Source |
|---|---|
| 1 | `azure.api_key` in config |
| 2 | `AZURE_API_KEY` environment variable |
| 3 | `DefaultAzureCredential` — managed identity, Azure CLI, VS Code, service principal |

In a corporate environment with managed identity or an active Azure CLI session, omit `api_key` entirely.

---

## CLI reference

### `staite run`

Generate a STATE.json snapshot.

```
staite run [OPTIONS]

Options:
  -c, --config PATH       Path to staite.yaml  [default: staite.yaml]
  -r, --root PATH         Project root to scan  [default: config file's directory]
  -l, --log-level LEVEL   DEBUG | INFO | WARNING | ERROR  [default: INFO]
  -v, --verbose           Shorthand for --log-level DEBUG
  --version               Show version and exit
```

### `staite serve`

Start the MCP server. Requires `pip install staite[vector]`.

```
staite serve [OPTIONS]

Options:
  --state PATH      Path to STATE.json to index. Repeat for multiple projects.  [required]
  --db PATH         Central ChromaDB directory  [default: ~/.staite/vector_db]
  -t, --transport   stdio | sse | http  [default: stdio]
  --host TEXT       Bind host for HTTP/SSE  [default: 0.0.0.0]
  -p, --port INT    Bind port for HTTP/SSE  [default: 8080]
  -l, --log-level   DEBUG | INFO | WARNING | ERROR  [default: INFO]
```

---

## Git integration

### Post-merge hook

Create `.git/hooks/post-merge` (chmod +x):

```bash
#!/bin/sh
staite run --config staite.yaml
git add .staite/STATE.json
git commit -m "chore: update project state"
```

### GitHub Actions

```yaml
name: Update project state

on:
  push:
    branches: [main]

jobs:
  staite:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e .
      - run: staite run --config staite.yaml
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      - uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "chore: update project state"
          file_pattern: ".staite/STATE.json .staite/cache.json .staite/synthesis.json"
```

---

## Extending

### Adding a language parser

Implement the `LanguageParser` protocol and register it:

```python
# staite/diagram/parsers/ruby.py
from pathlib import Path
import re

class RubyParser:
    extensions: list[str] = [".rb"]

    def extract_dependencies(self, filepath: Path, content: str) -> list[str]:
        return re.findall(r"require_relative ['\"]([^'\"]+)['\"]", content)
```

Then add it to `staite/diagram/parsers/__init__.py`:

```python
from staite.diagram.parsers.ruby import RubyParser

_DEFAULT_PARSERS = [
    PythonParser(),
    JavaScriptParser(),
    RubyParser(),
]
```

### Adding a provider

Implement the `LLMProvider` protocol and register it in `staite/providers/__init__.py`'s `create_provider()` factory.

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

All tests are fully unit-testable — no real API calls, no network.

---

## Cost

File descriptions use the cheapest available model (Haiku by default). The hash cache means only changed files are re-described on subsequent runs — the main cost is the first full scan and any run where enough files changed to trigger synthesis regeneration (`regen_threshold`).

Use `--verbose` to see cache hit/miss counts and token usage per run.
