"""
OpenAI Client Utilities
=======================

Helper functions and legacy wrappers for creating OpenAI clients that
automatically record interactions through Memoria's interception system.

The core Memoria package remains provider-agnostic. Similar modules can be
added for other providers without affecting the base code.
"""

from loguru import logger

from .openai_integration import register_memoria_instance


class MemoriaOpenAI:
    """DEPRECATED: Legacy OpenAI wrapper."""

    def __init__(self, memoria_instance, **kwargs):
        logger.warning(
            "MemoriaOpenAI is deprecated. Use memoria.enable() and the standard OpenAI client instead."
        )
        try:
            import openai

            self._openai = openai.OpenAI(**kwargs)

            # Register for automatic interception
            register_memoria_instance(memoria_instance)

            # Pass through all public attributes
            for attr in dir(self._openai):
                if not attr.startswith("_"):
                    setattr(self, attr, getattr(self._openai, attr))
        except ImportError as err:
            raise ImportError("OpenAI package required: pip install openai") from err


class MemoriaOpenAIInterceptor(MemoriaOpenAI):
    """DEPRECATED: Use automatic interception instead."""

    def __init__(self, memoria_instance, **kwargs):
        logger.warning(
            "MemoriaOpenAIInterceptor is deprecated. Use memoria.enable() then use OpenAI() client directly."
        )
        super().__init__(memoria_instance, **kwargs)


def create_openai_client(memoria_instance, provider_config=None, **kwargs):
    """
    Create a standard ``openai.OpenAI`` client that automatically records to
    the provided Memoria instance.

    Downstream integrations for other providers can follow this pattern by
    creating a ``<provider>_client`` module with a similar
    ``create_*_client`` helper.

    Args:
        memoria_instance: Memoria instance to record conversations to.
        provider_config: Optional ProviderConfig with default settings.
        **kwargs: Extra arguments forwarded to ``openai.OpenAI``.

    Returns:
        Configured OpenAI client instance.
    """
    try:
        import openai

        # Register the memoria instance for automatic interception
        register_memoria_instance(memoria_instance)

        # Use provider config if available, otherwise use kwargs
        if provider_config:
            client_kwargs = provider_config.get_openai_client_kwargs()
        else:
            client_kwargs = {}

        client_kwargs.update(kwargs)  # Allow kwargs to override config

        use_azure_client = client_kwargs.pop("_use_azure_client", False)

        # Create standard OpenAI client - it will be automatically intercepted
        if use_azure_client:
            azure_client_cls = getattr(openai, "AzureOpenAI", None)
            if azure_client_cls is None:
                logger.warning(
                    "Azure client requested but AzureOpenAI class not available; falling back to OpenAI"
                )
                client = openai.OpenAI(**client_kwargs)
            else:
                client = azure_client_cls(**client_kwargs)
        else:
            client = openai.OpenAI(**client_kwargs)

        logger.info("Created OpenAI client with automatic memoria recording")
        return client

    except ImportError as e:
        logger.error(f"Failed to import OpenAI: {e}")
        raise ImportError("OpenAI package required: pip install openai") from e
    except Exception as e:
        logger.error(f"Failed to create OpenAI client: {e}")
        raise


__all__ = ["MemoriaOpenAI", "MemoriaOpenAIInterceptor", "create_openai_client"]
