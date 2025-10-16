"""
Universal LLM Integration - Plug-and-Play Memory Recording

🎯 SIMPLE USAGE (RECOMMENDED):
Just call memoria.enable() and use ANY LLM library normally!

```python
from memoria import Memoria
from memoria.config import ConfigManager
import os

memoria = Memoria(...)
memoria.enable()  # 🎉 That's it!

# Pull model name from config or environment (LLM_MODEL)
model_name = ConfigManager().get_setting("agents.default_model", os.getenv("LLM_MODEL"))

# Now use ANY LLM library normally - all calls will be auto-recorded:

# LiteLLM (native callbacks)
from litellm import completion
completion(model=model_name or "gpt-4o-mini", messages=[...])  # ✅ Auto-recorded

# Direct OpenAI (auto-wrapping)
import openai
client = openai.OpenAI(api_key="...")
client.chat.completions.create(model=model_name or "gpt-4o-mini", messages=[...])  # ✅ Auto-recorded

# Direct Anthropic (auto-wrapping)
import anthropic
client = anthropic.Anthropic(api_key="...")
client.messages.create(model=model_name or "claude-3-opus", messages=[...])  # ✅ Auto-recorded
```

The universal system automatically detects and records ALL LLM providers
without requiring wrapper classes or complex setup.

Note:
    Set the default model in memoria.json under `agents.default_model` or export
    an environment variable such as LLM_MODEL/OPENAI_MODEL/etc. to override it.
"""

import os

from loguru import logger

from memoria.config import ConfigManager

# Legacy imports (all deprecated)
try:  # pragma: no cover - optional integrations
    from . import (
        anthropic_integration,
        litellm_integration,
        openai_client,
        openai_integration,
    )
except Exception as integration_import_error:
    logger.debug(
        f"Optional integrations unavailable during import: {integration_import_error}"
    )

__all__ = [
    # New interceptor classes (recommended)
    "MemoriaOpenAIInterceptor",
    # Wrapper classes for direct SDK usage (legacy)
    "MemoriaOpenAI",
    "MemoriaAnthropic",
    # Factory functions
    "create_openai_client",
    "setup_openai_interceptor",
    "get_default_model",
]


# Shared helper -------------------------------------------------------------


def get_default_model(env_var: str = "LLM_MODEL") -> str | None:
    """Return the configured default LLM model for Memoria integrations.

    Args:
        env_var: Optional environment variable name to consult as a fallback
            when the configuration value is unset.

    Returns:
        The configured model name, falling back to the environment variable if
        present, otherwise ``None``.
    """

    return ConfigManager().get_setting("agents.default_model", os.getenv(env_var))


# For backward compatibility, provide simple passthrough
try:
    from .anthropic_integration import MemoriaAnthropic
    from .openai_client import (
        MemoriaOpenAI,
        MemoriaOpenAIInterceptor,
        create_openai_client,
    )
    from .openai_integration import setup_openai_interceptor

    # But warn users about the better way for deprecated classes
    def __getattr__(name):
        if name == "MemoriaOpenAI":
            logger.warning(
                "🚨 MemoriaOpenAI wrapper class is deprecated!\n"
                "✅ NEW RECOMMENDED WAY: Use MemoriaOpenAIInterceptor or memoria.create_openai_client()"
            )
            return MemoriaOpenAI
        elif name == "MemoriaAnthropic":
            logger.warning(
                "🚨 MemoriaAnthropic wrapper class is deprecated!\n"
                "✅ NEW SIMPLE WAY: Use memoria.enable() and import anthropic normally"
            )
            return MemoriaAnthropic
        elif name in [
            "MemoriaOpenAIInterceptor",
            "create_openai_client",
            "setup_openai_interceptor",
        ]:
            # These are the new recommended classes/functions
            if name == "MemoriaOpenAIInterceptor":
                return MemoriaOpenAIInterceptor
            elif name == "create_openai_client":
                return create_openai_client
            elif name == "setup_openai_interceptor":
                return setup_openai_interceptor
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

except ImportError:
    # Wrapper classes not available, that's fine
    pass
