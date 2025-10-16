"""
Memoria LiteLLM Example.

Model configuration comes from ``memoria.json``; edit the
``agents.default_model`` field to change models.
"""

import json
from pathlib import Path

from litellm import completion

from memoria import Memoria

CONFIG_PATH = Path(__file__).resolve().parent / "memoria.json"
if not CONFIG_PATH.exists():
    CONFIG_PATH = Path(__file__).resolve().parents[1] / "memoria.json"
with open(CONFIG_PATH) as f:
    MODEL = json.load(f)["agents"]["default_model"]

print("Initializing Memoria with SQLite database...")
litellm_memory = Memoria(
    database_connect="sqlite:///litellm_demo.db",
    conscious_ingest=True,
    auto_ingest=True,
    # verbose=True,
)

print("Enabling memory tracking...")
litellm_memory.enable()

print("Memoria SQLite Demo - Chat with GPT-4o while memory is being tracked")
print("Type 'exit' or press Ctrl+C to quit")
print("-" * 50)

while 1:
    try:
        user_input = input("User: ")
        if not user_input.strip():
            continue

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        print("Processing your message with memory tracking...")
        response = completion(
            model=MODEL, messages=[{"role": "user", "content": user_input}]
        )
        print(f"AI: {response.choices[0].message['content']}")
        print()  # Add blank line for readability
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Error: {e}")
        continue
