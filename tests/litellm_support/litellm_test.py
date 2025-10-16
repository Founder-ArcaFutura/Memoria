import os
import sys
import time

import pytest

if not os.getenv("MEMORIA_RUN_INTEGRATION"):
    pytest.skip(
        "LiteLLM integration demo – enable MEMORIA_RUN_INTEGRATION=1 to execute.",
        allow_module_level=True,
    )

from litellm import completion

from memoria import Memoria

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.test_utils import get_test_model, load_inputs

litellm_memory = Memoria(
    database_connect="sqlite:///litellm_memory.db",
    conscious_ingest=True,
    verbose=True,
)

litellm_memory.enable()

# Load test inputs from JSON file
# Get the absolute path to test_inputs.json
json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_inputs.json")
# Optional: specify limit to load only first N inputs (e.g., limit=5)
test_inputs = load_inputs(json_path, limit=10)  # Load only first 10 inputs
# test_inputs = load_inputs(json_path)  # Load all inputs

MODEL = get_test_model()

for i, user_input in enumerate(test_inputs, 1):
    try:
        response = completion(
            model=MODEL, messages=[{"role": "user", "content": user_input}]
        )

        print(f"[{i}/{len(test_inputs)}] User: {user_input}")
        print(
            f"[{i}/{len(test_inputs)}] AI: {response.choices[0].message['content']}\n"
        )

        # Add small delay to avoid rate limiting
        time.sleep(1)

    except Exception as e:
        print(f"[{i}/{len(test_inputs)}] Error: {e}")
        print("Waiting 60 seconds before continuing...")
        time.sleep(60)
