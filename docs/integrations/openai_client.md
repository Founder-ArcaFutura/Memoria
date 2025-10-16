# OpenAI Client Helper

The `memoria.integrations.openai_client` module contains optional helpers for
working with the OpenAI Python SDK. It provides:

- `create_openai_client` – returns a standard `openai.OpenAI` client wired to
  automatically record conversations through Memoria's interceptor system.
- Legacy wrappers (`MemoriaOpenAI` and `MemoriaOpenAIInterceptor`) preserved for
  backward compatibility.

These utilities live outside of the core Memoria classes so that the package
remains provider‑agnostic. Downstream projects can follow this pattern to add
modules such as `anthropic_client.py` or `vertexai_client.py` with their own
`create_*_client` helpers.

```python
from memoria import Memoria

mem = Memoria(api_key="sk-...")
mem.enable()

# Create a client that records automatically
client = mem.create_openai_client()
response = client.chat.completions.create(
    model=mem.model,
    messages=[{"role": "user", "content": "Hello!"}],
)
```

See `memoria.integrations.openai_client` for the full implementation and use it
as a reference when integrating other providers.
