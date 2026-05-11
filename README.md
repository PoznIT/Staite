# StAIte

Generate a dense, AI-readable snapshot of your project and use it as context in Claude chat — without burning tokens re-explaining your codebase every conversation.

## The problem

Claude Code is powerful but expensive in tokens. Claude chat is cheap but stateless — every new conversation starts from zero, and maintaining context manually is tedious.

StAIte solves this by scanning your project on every merge to main and producing a single `STATE.xml` file you paste into your Claude project instructions. The AI then knows your project structure, what every file does, how components connect, and what conventions you follow — before you type a single word.

## How it works

1. **Scan** the filesystem (no AI, just pathspec pattern matching)
2. **Describe** each file with 2–6 sentences via a cheap AI model — skipping files unchanged since the last run (SHA-256 hash cache)
3. **Diagram** the directory tree and intra-project dependencies as Mermaid
4. **Synthesise** use-case scenarios and coding conventions (regenerated only when enough files changed)
5. **Assemble** everything into a Claude-optimised XML file

---

## Installation

Requires Python 3.11+.

```bash
git clone <this repo>
cd StAIte
pip install -e .
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
```

**3. Run:**

```bash
python -m staite --config staite.yaml
```

The state file is written to `.staite/STATE.xml` (configurable). Paste its contents into your Claude project instructions.

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

# Free-form context for the AI — architecture decisions, conventions,
# entry points, things that aren't obvious from the code
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
# claude-sonnet-4-6 or claude-opus-4-6 for higher quality on complex codebases.
model: claude-haiku-4-5-20251001

# Regenerate use-cases and conventions only if this fraction of files changed.
# 0.2 = regenerate when >20% of files were modified since last run.
regen_threshold: 0.2

output: .staite/STATE.xml
cache: .staite/cache.json
synthesis_cache: .staite/synthesis.json
```

---

## Providers

### Anthropic (default)

```yaml
provider: anthropic
```

Requires the `ANTHROPIC_API_KEY` environment variable.

### Azure AI Foundry

```bash
pip install staite[azure]
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

In a corporate environment with managed identity or an active Azure CLI session, omit `api_key` entirely and `DefaultAzureCredential` handles auth automatically.

---

## Output format

`STATE.xml` uses Anthropic-recommended XML tags so Claude parses sections reliably:

```xml
<project_state>
  <metadata>        name, generation timestamp, file count</metadata>
  <instructions>    your free-form context from config</instructions>
  <use_cases>       AI-generated: what this project does and key flows</use_cases>
  <conventions>     [User-defined] your conventions + [AI-inferred] detected patterns</conventions>
  <file_tree>       indented directory tree</file_tree>
  <architecture>    Mermaid diagram: directory structure + dependency edges</architecture>
  <files>           one <file path="..."> per scanned file with its description</files>
</project_state>
```

Sections are ordered so Claude reads the highest-signal content first.

---

## CLI reference

```
python -m staite [OPTIONS]

Options:
  -c, --config PATH       Path to staite.yaml  [default: staite.yaml]
  -r, --root PATH         Project root to scan  [default: config file's directory]
  -l, --log-level LEVEL   DEBUG | INFO | WARNING | ERROR  [default: INFO]
  -v, --verbose           Shorthand for --log-level DEBUG
  --version               Show version and exit
```

---

## Git integration

### Post-merge hook

Create `.git/hooks/post-merge` (chmod +x):

```bash
#!/bin/sh
python -m staite --config staite.yaml
git add .staite/STATE.xml
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
      - run: python -m staite --config staite.yaml
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      - uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "chore: update project state"
          file_pattern: ".staite/STATE.xml .staite/cache.json .staite/synthesis.json"
```

---

## Extending

### Adding a language parser

Dependency parsing is language-agnostic by design. To add support for a new language, implement the `LanguageParser` protocol and register it:

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

_DEFAULT_PARSERS: list[LanguageParser] = [
    PythonParser(),
    JavaScriptParser(),
    RubyParser(),   # add here
]
```

That's it — no other changes needed.

### Adding a provider

Implement the `LLMProvider` protocol:

```python
# staite/providers/my_provider.py
class MyProvider:
    async def complete(self, prompt: str, max_tokens: int) -> str:
        ...

    async def __aenter__(self) -> "MyProvider":
        return self

    async def __aexit__(self, *args: object) -> None:
        ...
```

Register it in `staite/providers/__init__.py`'s `create_provider()` factory.

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run a specific module's tests
pytest tests/test_describer.py -v
```

All tests are fully unit-testable — no real API calls, no network.

---

## Cost

File descriptions use the cheapest available model (Haiku by default). The hash cache means only changed files are re-described on subsequent runs — the main cost driver is the first full scan and any run where enough files changed to trigger synthesis regeneration (`regen_threshold`).

Actual cost depends on your project's file count, average file size, and whether synthesis regenerates. Check the [Anthropic pricing page](https://www.anthropic.com/pricing) for current token rates and use `--verbose` to see how many files were cache misses on each run.
