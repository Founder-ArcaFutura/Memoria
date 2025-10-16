"""
Basic Memoria Usage Example
Simple demonstration of conscious ingestion and context injection.

Model configuration is read from ``memoria.json``. Update the
``agents.default_model`` field in that file to use your preferred model.
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from litellm import completion

from memoria import Memoria

CONFIG_PATH = Path(__file__).resolve().parent / "memoria.json"
if not CONFIG_PATH.exists():
    CONFIG_PATH = Path(__file__).resolve().parents[1] / "memoria.json"
with open(CONFIG_PATH) as f:
    MODEL = json.load(f)["agents"]["default_model"]

load_dotenv()


def main():
    print("🧠 Memoria - AI Memory with Conscious Ingestion")
    print("=" * 55)

    # Initialize your workspace memory with conscious ingestion
    office_work = Memoria(
        database_connect="sqlite:///office_memory.db",
        conscious_ingest=True,  # 🔥 Enable AI-powered background analysis
        verbose=True,  # Show what's happening behind the scenes
        openai_api_key=None,  # Uses OPENAI_API_KEY from environment
    )

    # Enable memory recording
    office_work.enable()
    print("✅ Memory enabled - all conversations will be recorded!")

    # Store an explicit memory with emotional intensity (saved in the
    # processed_data JSON field)
    office_work.store_memory(
        anchor="mood",
        text="I'm thrilled about this project!",
        tokens=6,
        emotional_intensity=0.9,
        symbolic_anchors=["excitement"],
    )

    # First conversation - establishing context
    print("\n--- First conversation ---")
    response1 = completion(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "I'm working on a FastAPI project with PostgreSQL database",
            }
        ],
    )
    print(f"Assistant: {response1.choices[0].message.content}")

    # Second conversation - memory automatically provides context
    print("\n--- Second conversation (with memory context) ---")
    response2 = completion(
        model=MODEL,
        messages=[
            {"role": "user", "content": "Help me write database connection code"}
        ],
    )
    print(f"Assistant: {response2.choices[0].message.content}")

    # Third conversation - showing preference memory
    print("\n--- Third conversation (preferences remembered) ---")
    response3 = completion(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "I prefer clean, well-documented code with type hints",
            }
        ],
    )
    print(f"Assistant: {response3.choices[0].message.content}")

    # Fourth conversation - memory knows your preferences
    print("\n--- Fourth conversation (preferences applied) ---")
    response4 = completion(
        model=MODEL,
        messages=[{"role": "user", "content": "Show me how to create a user model"}],
    )
    print(f"Assistant: {response4.choices[0].message.content}")

    print("\n🎉 That's it! Your AI now remembers your:")
    print("  • Tech stack (FastAPI, PostgreSQL)")
    print("  • Coding preferences (clean code, type hints)")
    print("  • Project context (user models, database connections)")
    print("\n🧠 With conscious_ingest=True:")
    print("  • Background analysis will identify essential information")
    print("  • Key facts automatically promoted for instant access")
    print("  • Context injection gets smarter over time")
    print("\nNo more repeating context - just chat naturally!")


if __name__ == "__main__":
    main()
