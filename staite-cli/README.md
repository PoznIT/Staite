# StAIte CLI

## Prerequisites

- Python 3.14
- An AI provider in the following
  - Azure OpenAI
  - Anthropic
  - Ollama (local)

## Run

### Commands
```bash
# Generate the staite file, default path: .staite/STATE.json
python3 cli.py run 
# Push the staite file to the server in the config
python3 cli.py update <path_to_staite_file>
```

### Example config for local run
> [!IMPORTANT]
> This config works with local ollama instance with `qwen3:8b` installed

```yaml
project_name: staite-cli
provider: ollama

ollama:
  url: http://localhost:11434
  model: qwen3:8b

include:
  - src/
  - pyproject.toml
  - cli.py

exclude:
  - venv/**
  - tests/**
  - staite.egg-inf/**
  - .idea/**
  - "**/__pycache__/**"
  - "**/*.pyc"
  - "**/.DS_Store"
  - node_modules/**
  - dist/**
  - build/**
  - "**/*.min.js"
  - "**/*.lock"

instructions: |
  This project is meant to generate a `STATE.json` file. 
  The user will use the cli in his own project to generate 
  - filesystem tree
  - usecase scenario of his project
  - description of each files
  - coding conventions

output: .staite/STATE.json
cache: .staite/cache.json
```