# StAIte MCP

## Local run 
### Prerequisites

- Ollama CLI
- Docker compose
- Python 3.14

### Ports
| Port   | App            | Description                                         |
|--------|----------------|-----------------------------------------------------|
| 8000   | Chroma DB      | Vector DB                                           |
| 114343 | Ollama service | LLM local service for embedding and text generation |
| 8080   | MCP server     | The actual served staite MCP                        |

### Installation

```bash
ollama pull nomic-embed-text # pull embedding model
docker-compose up --build -d # Run chroma DB
python3 main.py -c <path to conf - default: ./config.yml> # serve the app
# Optionally run the mcp inspector
 npx @modelcontextprotocol/inspector
```

By default, the config will be:

```yml
log_level: INFO

chroma:
  host: localhost
  port: 8000

ollama:
  url: http://localhost:11434
  model: nomic-embed-text

server:
  transport: http
  host: 0.0.0.0
  port: 8080
```