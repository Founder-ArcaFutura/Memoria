"""
Memoria AgentOps Integration Example
==================================

Track and monitor Memoria memory operations with [AgentOps](https://www.agentops.ai/)

- Memory Recording: Track when conversations are automatically captured and stored
- Context Injection: Monitor how memory is automatically added to LLM context
- Conversation Flow: Understand the complete dialogue history across sessions
- Memory Effectiveness: Analyze how historical context improves response quality
- Performance Impact: Track latency and token usage from memory operations
- Error Tracking: Identify issues with memory recording or context retrieval

AgentOps automatically instruments Memoria to provide complete observability
of your memory operations.

Installation
------------

```bash
pip install -e .
pip install agentops openai python-dotenv
```

Requirements
-----------
- OPENAI_API_KEY environment variable
- AGENTOPS_API_KEY environment variable

The model for LLM calls is loaded from ``memoria.json``. Adjust
``agents.default_model`` to change it.
"""

import json
from pathlib import Path

import agentops
from openai import OpenAI

from memoria import Memoria

CONFIG_PATH = Path(__file__).resolve().parent / "memoria.json"
if not CONFIG_PATH.exists():
    CONFIG_PATH = Path(__file__).resolve().parents[1] / "memoria.json"
with open(CONFIG_PATH) as f:
    MODEL = json.load(f)["agents"]["default_model"]

# Start a trace to group related operations
agentops.start_trace("memoria_conversation_flow", tags=["memoria_memory_example"])

try:
    # Initialize OpenAI client
    openai_client = OpenAI()

    # Initialize Memoria with conscious ingestion enabled
    # AgentOps tracks the memory configuration
    memoria = Memoria(
        database_connect="sqlite:///agentops_example.db",
        conscious_ingest=True,
        auto_ingest=True,
    )

    memoria.enable()

    # First conversation - AgentOps tracks LLM call and memory recording
    response1 = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "I'm working on a Python FastAPI project"}
        ],
    )

    print("Assistant:", response1.choices[0].message.content)

    # Second conversation - AgentOps tracks memory retrieval and context injection
    response2 = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Help me add user authentication"}],
    )

    print("Assistant:", response2.choices[0].message.content)
    print("💡 Notice: Memoria automatically provided FastAPI project context!")

    # End trace - AgentOps aggregates all operations
    agentops.end_trace(end_state="success")

except Exception:
    agentops.end_trace(end_state="error")
