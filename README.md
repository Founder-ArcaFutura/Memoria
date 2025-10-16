![Arca Futura](./Arca%20Futura%20-%20Corporate%20Logo%20v%201.0.png)

# Memoria



Memoria is an open-source, auditable memory system for Large Language Models (LLMs). It gives your AI agents a persistent, queryable "working memory" that goes far beyond the limits of a standard context window.

This project is designed for power users and developers who want to give their agents a reliable, long-term memory that they can easily host on their own machine. The goal is to provide a "minimal setup" experience, allowing you to get a sophisticated memory server up and running with very little fuss.

---

### A Note on Attribution

This project, **Memoria**, is a fork of the original **Memori** project by [GibsonAI](https://github.com/GibsonAI/memori). We are immensely grateful for their foundational work. This version, developed by **Arca Futura**, builds upon that foundation by adding new features like 3D spatial memory, a web UI, and heuristic-based ingestion. We believe in open source and are proud to contribute back to the community.

---

## 🚀 Quick Start: Minimal Setup

Get your own Memoria server running in just a few minutes.

### 1. Prerequisites
- Python 3.10 or newer.

### Installation
```bash
git clone https://github.com/Founder-ArcaFutura/memoria_staging.git
cd memoria_staging
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Usage
```bash
memoria --help
```

### Examples
```bash
# Run the Researcher Agent demo
python demos/researcher_agent/app.py

# Or the Travel Planner
python demos/travel_planner/app.py
```

### Documentation

Build locally with:
```bash
mkdocs serve
```

or visit the hosted docs at (insert GitHub Pages URL here).

### 3. Configuration
Memoria is configured with a simple YAML file.

We provide several example configurations in the `configurations/` directory to help you get started:

*   **`personal_default.yaml`**: The simplest way to get started. It uses a local SQLite database and is perfect for personal use.
*   **`personal_with_postgres.yaml`**: For users who want a more robust, production-ready setup using a PostgreSQL database.
*   **`team_experimental.yaml`**: Demonstrates how to enable the experimental team-aware features for multi-user collaboration.

To use one of these, copy it to `config.yaml` and fill in your details:
```bash
cp configurations/personal_default.yaml config.yaml
```

Once you've copied your preferred configuration, open `config.yaml` and add your LLM provider's API key. For example, for OpenAI:
```yaml
# In config.yaml
agents:
  openai_api_key: "sk-..."
```

Finally, for security, set a unique API key for your server. This is done with an environment variable:
```bash
export MEMORIA_API_KEY='a-very-secret-key-that-you-create'
```
The server will require this key for all API requests.

### 4. Run the Server
Launch the server using a production-ready tool like Gunicorn:
```bash
gunicorn --reload -w 4 -b 0.0.0.0:8080 "memoria_server.api.app_factory:create_app()"
```
Your Memoria server is now running on `http://localhost:8080`.

### 5. Connect to your LLM (e.g., a Custom GPT)
To give your agent access to its new memory, you need to expose your local server to the internet. [ngrok](https://ngrok.com/) is a great tool for this.

Once `ngrok` is installed and configured, run:
```bash
ngrok http 8080
```
`ngrok` will give you a public URL (e.g., `https://random-string.ngrok-free.app`).

You can now use this URL to create a new Action for a Custom GPT. Use the public `ngrok` URL as the server and import the API schema from `http://localhost:8080/openapi.json` (served by the `/openapi.json` route in the `utility` blueprint). Don't forget to add your `MEMORIA_API_KEY` as an authentication header!

### API Reference

The running server exposes the full OpenAPI specification at `GET /openapi.json`. The endpoint is provided by the `utility` blueprint and simply streams the repository's `openapi.json` file, making it easy to keep custom clients and integrations in sync with the deployed server.

## 📚 Examples

All runnable examples now live in the [`examples/`](./examples) directory so the project root stays focused on supported assets. Highlights include:

- [`examples/basic_usage.py`](./examples/basic_usage.py) – the minimal getting-started flow.
- [`examples/memoria_example.py`](./examples/memoria_example.py) – a conscious-ingest walkthrough with interactive prompts.
- [`examples/auto_ingest_example.py`](./examples/auto_ingest_example.py) – dynamic retrieval with automated context injection.
- [`examples/memory_retrieval_example.py`](./examples/memory_retrieval_example.py) – function-calling integration with the memory tool.

## 🧠 Personal Ingestion Mode

Memoria includes a "personal" ingestion mode for capturing user-specific context directly into long-term storage. This mode is ideal when you want memories to skip the short-term staging queue while still tracking spatial metadata and attached source documents.

### Enabling personal ingestion

1. Update `config.yaml` so that the memory service starts in personal mode:

   ```yaml
   memory:
     ingest_mode: personal
     personal_documents_enabled: true
   ```

   You can achieve the same result with environment variables:

   ```bash
   export MEMORIA_MEMORY_INGEST_MODE=personal
   export MEMORIA_MEMORY_PERSONAL_DOCUMENTS_ENABLED=true
   ```

2. Restart the server so the new ingest mode and document support flags are applied.

With these settings in place, the `/memory` endpoint accepts personal payloads by default. You can also opt-in on a per-request basis by including `"ingest_mode": "personal"` in the JSON body.

### Sending a personal ingestion request

The personal pipeline requires a `chat_id` so the captured memory can be linked back to the originating conversation. Optional `documents` allow you to attach supporting materials that will be stored alongside the memory.

```bash
curl -X POST "http://localhost:8080/memory" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $MEMORIA_API_KEY" \
  -d '{
        "anchor": "daily-reflection",
        "text": "Captured directly from my private chat.",
        "tokens": 42,
        "timestamp": "2025-01-01T18:30:00Z",
        "x_coord": 0.0,
        "y_coord": 1.5,
        "z_coord": -0.5,
        "symbolic_anchors": ["personal", "journal"],
        "chat_id": "chat-session-123",
        "metadata": {"topic": "journaling"},
        "documents": [
          {
            "document_id": "doc-42",
            "title": "Chat transcript",
            "url": "https://example.com/transcript"
          }
        ],
        "ingest_mode": "personal"
      }'
```

### How personal ingestion differs from the standard pipeline

- **Direct long-term storage:** personal payloads bypass the short-term staging heuristics entirely.
- **Conversation linkage:** the supplied `chat_id` is recorded as the memory's originating conversation so later retrieval can follow the same thread.
- **Document preservation:** any attached `documents` are normalized and stored with the memory, enabling downstream citation or review workflows.
- **Spatial metadata retention:** coordinates and symbolic anchors are still written to the spatial metadata table, ensuring the memory participates in spatial queries.

## ✨ Features

### Core Features
These features are stable and ready for use.
- **3D Spatial Memory:** Memories are mapped in a 3D space (`time`, `privacy`, `abstraction`), allowing for nuanced, context-aware retrieval.
- **Heuristic Ingestion & Fallback:** Memoria can intelligently process and store memories without relying on an LLM, making it suitable for air-gapped or cost-sensitive deployments.
- **Web Dashboard UI:** A built-in web interface to inspect memories, view system settings, and understand the state of your agent's memory.
- **Symbolic Anchors & Clustering:** Group related memories into clusters using symbolic tags for high-level conceptual search.

### Experimental Features
These features are included but are still under development. They can be enabled via the `config.yaml` file.
- **Team-Aware Deployments:** Allows multiple users or agents to collaborate with shared memory spaces while maintaining privacy.
- **Secure Database Connections:** Configure Memoria to connect to a production database like PostgreSQL instead of the default SQLite.
- **Advanced Governance Framework:** Define fine-grained retention policies, privacy floors, and escalation rules for memories.
- **Vector Search & Clustering:** Use vector embeddings for semantic search and clustering, in addition to the standard heuristics.
- **Cross-Instance Sync:** Use Redis or PostgreSQL to synchronize memory across multiple Memoria instances.
- **Command-Line Interface (CLI):** A set of tools for bootstrapping, data migration, and managing your Memoria instance from the terminal.

## 🚀 Advanced Usage

For more complex deployments, including information on running Memoria in a team environment or with a production database, see the [Advanced Usage](docs/advanced_usage) documentation.

## How It Works
Memoria ingests information, processes it through a pipeline, and stores it in a structured, queryable format.
1.  **Recording:** All interactions can be recorded automatically.
2.  **Processing:** Information is processed to extract entities, assign spatial coordinates, and score for importance. This can be done via heuristics or by calling an LLM.
3.  **Storage:** Memories are stored in a local SQL database (SQLite by default), making them easy to inspect and audit.

## Contributing
We welcome contributions! Please read the [contribution guidelines](CONTRIBUTING.md) to get started.

## License
Memoria is released under the [Apache 2.0 License](LICENSE).
