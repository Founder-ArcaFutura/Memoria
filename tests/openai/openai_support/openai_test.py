import os
import sys

import pytest
from openai import OpenAI

from memoria import Memoria

pytest.skip("Skipping interactive OpenAI demo.", allow_module_level=True)


# Fix imports to work from any directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# --- Your code starts here ---

# Initialize Memoria
# Will use OPENAI_API_KEY from environment by default
memoria = Memoria()
memoria.enable()


# Use Memoria's pre-configured client for interception
client = OpenAI()

# Use Memoria's built-in client for interception
# client = memoria.create_openai_client()

# --- Example usage ---
response = client.chat.completions.create(
    model=memoria.model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
)

print(response.choices[0].message.content)
