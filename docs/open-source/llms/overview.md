# Supported LLM Providers

Memoria provides universal integration with any LLM provider through multiple integration approaches. Below is a comprehensive table of tested and supported LLM providers with links to working examples.

## Supported LLM Providers

| Provider | Example Link |
|----------|--------------|
| **OpenAI** | [OpenAI Example](https://github.com/Founder-ArcaFutura/Memoria/tree/main/examples/supported_llms/openai_example.py) |
| **Azure OpenAI** | [Azure Example](https://github.com/Founder-ArcaFutura/Memoria/tree/main/examples/supported_llms/azure_example.py) |
| **LiteLLM** | [LiteLLM Example](https://github.com/Founder-ArcaFutura/Memoria/tree/main/examples/supported_llms/litellm_example.py) |
| **Ollama** | [Ollama Example](https://github.com/Founder-ArcaFutura/Memoria/tree/main/examples/supported_llms/ollama_example.py) |
| **Anthropic Claude** | [Provider Config](https://github.com/Founder-ArcaFutura/Memoria/tree/main/memoria/core/providers.py) |
| **Google Gemini** | [Provider Config](https://github.com/Founder-ArcaFutura/Memoria/tree/main/memoria/core/providers.py) |
| **Any OpenAI-Compatible** | [Provider Config](https://github.com/Founder-ArcaFutura/Memoria/tree/main/memoria/core/providers.py) |

## Required Environment Variables

The Memoria runtime reads standard environment variables for each provider during
initialization:

- **OpenAI / Azure OpenAI**
  - `OPENAI_API_KEY`
  - `AZURE_OPENAI_API_KEY`
  - `OPENAI_API_TYPE`, `OPENAI_BASE_URL`, and other Azure deployment fields as needed
- **Anthropic Claude**
  - `ANTHROPIC_API_KEY`
  - Optional: `ANTHROPIC_BASE_URL` and `ANTHROPIC_MODEL`
- **Google Gemini**
  - `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
  - Optional: `GEMINI_MODEL`
  - Install optional dependencies: `pip install -e ".[integrations]"`

## Community Contributions

Missing a provider? Memoria's universal integration approach means most LLM providers work out of the box. If you need specific support for a new provider:

1. Check if it's OpenAI-compatible (most are)
2. Try the universal integration first
3. Open an issue if you need custom integration support

All examples and integrations are maintained in the [GitHub repository](https://github.com/Founder-ArcaFutura/Memoria) with regular updates as new providers and frameworks emerge.
