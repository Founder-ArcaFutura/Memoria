"""Helper utilities for configuring provider clients and LLM agents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from loguru import logger

from .providers import (
    ProviderRegistry,
    ProviderRouteMetadata,
    ProviderType,
    RegisteredProvider,
    TaskRouteSpec,
)


@dataclass(slots=True)
class ProviderSetupInputs:
    """Input parameters required to configure provider and agent state."""

    default_model: str
    provider_config: Any | None = None
    api_type: str | None = None
    base_url: str | None = None
    azure_endpoint: str | None = None
    api_key: str | None = None
    openai_api_key: str | None = None
    azure_deployment: str | None = None
    api_version: str | None = None
    azure_ad_token: str | None = None
    organization: str | None = None
    project: str | None = None
    model: str | None = None
    sovereign_ingest: bool = False
    conscious_ingest: bool = False
    auto_ingest: bool = False
    enable_short_term: bool = True
    use_lightweight_conscious_ingest: bool = False
    anthropic_config: Any | None = None
    anthropic_api_key: str | None = None
    anthropic_base_url: str | None = None
    anthropic_model: str | None = None
    gemini_config: Any | None = None
    gemini_api_key: str | None = None
    gemini_model: str | None = None
    task_routes: Mapping[str, TaskRouteSpec] | None = None


@dataclass(slots=True)
class ProviderSetupResult:
    """Return value containing configured provider, agents, and clients."""

    provider_config: Any | None
    api_key: str
    model: str
    provider_registry: ProviderRegistry | None
    task_routes: dict[str, TaskRouteSpec]
    anthropic_config: Any | None
    anthropic_client: Any | None
    anthropic_async_client: Any | None
    gemini_config: Any | None
    gemini_client: Any | None
    memory_agent: Any | None
    search_engine: Any | None
    conscious_agent: Any | None
    conscious_ingest: bool
    auto_ingest: bool


def setup_provider_components(inputs: ProviderSetupInputs) -> ProviderSetupResult:
    """Configure provider clients and initialize memory/search agents."""

    provider, api_key = _configure_primary_provider(inputs)

    model = inputs.model or getattr(provider, "model", None) or inputs.default_model
    if provider and not getattr(provider, "model", None):
        provider.model = model

    registry = ProviderRegistry()
    task_routes = dict(inputs.task_routes or {})

    (
        anthropic_config,
        anthropic_client,
        anthropic_async_client,
    ) = _configure_anthropic(inputs)
    gemini_config, gemini_client = _configure_gemini(inputs)

    _register_primary_provider(
        registry,
        provider_config=provider,
        api_key=api_key,
        model=model,
        inputs=inputs,
    )
    _register_optional_providers(
        registry,
        anthropic_config=anthropic_config,
        anthropic_client=anthropic_client,
        anthropic_async_client=anthropic_async_client,
        gemini_config=gemini_config,
        gemini_client=gemini_client,
    )

    if task_routes:
        registry.set_task_routes(task_routes)

    (
        memory_agent,
        search_engine,
        conscious_agent,
        conscious_ingest,
        auto_ingest,
    ) = _initialize_agents(
        provider=provider,
        api_key=api_key,
        model=model,
        inputs=inputs,
        provider_registry=registry,
        task_routes=task_routes,
    )

    return ProviderSetupResult(
        provider_config=provider,
        api_key=api_key,
        model=model,
        provider_registry=registry,
        task_routes=task_routes,
        anthropic_config=anthropic_config,
        anthropic_client=anthropic_client,
        anthropic_async_client=anthropic_async_client,
        gemini_config=gemini_config,
        gemini_client=gemini_client,
        memory_agent=memory_agent,
        search_engine=search_engine,
        conscious_agent=conscious_agent,
        conscious_ingest=conscious_ingest,
        auto_ingest=auto_ingest,
    )


def _configure_primary_provider(
    inputs: ProviderSetupInputs,
) -> tuple[Any | None, str]:
    """Construct the ProviderConfig instance using the supplied overrides."""

    resolved_api_key = inputs.api_key or inputs.openai_api_key or ""

    provider = inputs.provider_config
    if provider is not None:
        logger.info(
            "Using provided ProviderConfig with api_type: {}",
            getattr(provider, "api_type", "unknown"),
        )
    elif any([inputs.api_type, inputs.base_url, inputs.azure_endpoint]):
        try:
            from .providers import ProviderConfig

            if inputs.azure_endpoint:
                provider = ProviderConfig.from_azure(
                    api_key=resolved_api_key,
                    azure_endpoint=inputs.azure_endpoint,
                    azure_deployment=inputs.azure_deployment,
                    api_version=inputs.api_version,
                    azure_ad_token=inputs.azure_ad_token,
                    model=inputs.model or inputs.default_model,
                )
                logger.info("Using explicitly configured Azure OpenAI provider")
            elif inputs.base_url:
                provider = ProviderConfig.from_custom(
                    base_url=inputs.base_url,
                    api_key=resolved_api_key,
                    model=inputs.model or inputs.default_model,
                )
                logger.info(
                    "Using explicitly configured custom provider: {}",
                    inputs.base_url,
                )
            else:
                provider = ProviderConfig.from_openai(
                    api_key=resolved_api_key,
                    organization=inputs.organization,
                    project=inputs.project,
                    model=inputs.model or inputs.default_model,
                )
                logger.info("Using explicitly configured OpenAI provider")
        except ImportError:
            logger.warning("ProviderConfig not available, using basic configuration")
            provider = None
    else:
        try:
            from .providers import ProviderConfig

            provider = ProviderConfig.from_openai(
                api_key=resolved_api_key,
                organization=inputs.organization,
                project=inputs.project,
                model=inputs.model or inputs.default_model,
            )
            logger.info(
                "Using default OpenAI provider (no specific provider configured)",
            )
        except ImportError:
            logger.warning("ProviderConfig not available, using basic configuration")
            provider = None

    if provider and hasattr(provider, "api_key"):
        provider_api_key = getattr(provider, "api_key", None)
        if provider_api_key:
            resolved_api_key = provider_api_key

    return provider, resolved_api_key


def _register_primary_provider(
    registry: ProviderRegistry,
    *,
    provider_config: Any | None,
    api_key: str,
    model: str,
    inputs: ProviderSetupInputs,
) -> None:
    """Register the primary OpenAI-compatible provider with the registry."""

    provider_type = _detect_provider_type(provider_config, inputs)
    aliases: list[str] = []
    if provider_type in {
        ProviderType.OPENAI,
        ProviderType.OPENAI_COMPATIBLE,
        ProviderType.AZURE,
        ProviderType.CUSTOM,
    }:
        aliases.extend(["openai", "primary"])

    def _make_client() -> Any:
        if provider_config is not None:
            return provider_config.create_client()
        import openai

        return openai.OpenAI(api_key=api_key)

    def _make_async_client() -> Any:
        if provider_config is not None:
            return provider_config.create_async_client()
        import openai

        return openai.AsyncOpenAI(api_key=api_key)

    provider = RegisteredProvider(
        name=_provider_name_from_type(provider_type),
        provider_type=provider_type,
        model=model,
        metadata=ProviderRouteMetadata(
            preferred_tasks=(
                "memory_ingest",
                "search_planning",
                "default",
            ),
            cost_profile="standard",
            supports_structured_outputs=True,
        ),
        client_factory=_make_client,
        async_client_factory=_make_async_client,
    )

    registry.register(provider, aliases=aliases, is_primary=True)


def _register_optional_providers(
    registry: ProviderRegistry,
    *,
    anthropic_config: Any | None,
    anthropic_client: Any | None,
    anthropic_async_client: Any | None,
    gemini_config: Any | None,
    gemini_client: Any | None,
) -> None:
    """Register optional provider clients (Anthropic, Gemini)."""

    if anthropic_config is not None:
        provider = RegisteredProvider(
            name="anthropic",
            provider_type=ProviderType.ANTHROPIC,
            model=getattr(anthropic_config, "model", None),
            metadata=ProviderRouteMetadata(
                preferred_tasks=("search_planning", "memory_ingest"),
                cost_profile="premium",
                supports_structured_outputs=False,
            ),
            client_factory=(
                anthropic_config.create_client
                if hasattr(anthropic_config, "create_client")
                else None
            ),
            async_client_factory=(
                anthropic_config.create_async_client
                if hasattr(anthropic_config, "create_async_client")
                else None
            ),
            client_instance=anthropic_client,
            async_client_instance=anthropic_async_client,
        )
        registry.register(provider, aliases=["claude", "anthropic"], is_primary=False)

    if gemini_config is not None:
        provider = RegisteredProvider(
            name="gemini",
            provider_type=ProviderType.GOOGLE_GEMINI,
            model=getattr(gemini_config, "model", None),
            metadata=ProviderRouteMetadata(
                preferred_tasks=("search_planning",),
                cost_profile="high",
                supports_structured_outputs=False,
            ),
            client_factory=(
                gemini_config.create_client
                if hasattr(gemini_config, "create_client")
                else None
            ),
            async_client_factory=(
                gemini_config.create_async_client
                if hasattr(gemini_config, "create_async_client")
                else None
            ),
            client_instance=gemini_client,
            async_client_instance=None,
        )
        registry.register(provider, aliases=["google", "gemini"], is_primary=False)


def _detect_provider_type(
    provider_config: Any | None, inputs: ProviderSetupInputs
) -> ProviderType:
    """Infer the provider type from configuration overrides."""

    api_type = None
    if provider_config is not None:
        api_type = getattr(provider_config, "api_type", None)
    if api_type is None:
        api_type = inputs.api_type

    if api_type:
        normalized = api_type.strip().lower()
        if normalized == "azure":
            return ProviderType.AZURE
        if normalized in {"custom", "openai_compatible"}:
            return ProviderType.OPENAI_COMPATIBLE
        if normalized == "openai":
            return ProviderType.OPENAI

    return ProviderType.OPENAI


def _provider_name_from_type(provider_type: ProviderType) -> str:
    if provider_type == ProviderType.AZURE:
        return "azure"
    if provider_type == ProviderType.CUSTOM:
        return "custom"
    if provider_type == ProviderType.OPENAI_COMPATIBLE:
        return "openai-compatible"
    return provider_type.value


def _configure_anthropic(
    inputs: ProviderSetupInputs,
) -> tuple[Any | None, Any | None, Any | None]:
    """Configure Anthropic client instances when requested."""

    config = None
    client = None
    async_client = None

    if inputs.anthropic_config or inputs.anthropic_api_key or inputs.anthropic_base_url:
        try:
            from .providers import AnthropicConfig

            config = inputs.anthropic_config or AnthropicConfig(
                api_key=inputs.anthropic_api_key,
                base_url=inputs.anthropic_base_url,
                model=inputs.anthropic_model,
            )

            if inputs.anthropic_model and getattr(config, "model", None) in {None, ""}:
                config.model = inputs.anthropic_model
            if inputs.anthropic_api_key and getattr(config, "api_key", None) in {
                None,
                "",
            }:
                config.api_key = inputs.anthropic_api_key
            if inputs.anthropic_base_url and getattr(config, "base_url", None) in {
                None,
                "",
            }:
                config.base_url = inputs.anthropic_base_url

            try:
                client = config.create_client()
                async_client = None
                if hasattr(config, "create_async_client"):
                    try:
                        async_client = config.create_async_client()
                    except NotImplementedError:
                        async_client = None
                logger.info("Anthropic provider configured successfully")
            except ImportError as exc:
                logger.warning(
                    "Anthropic SDK is not installed; skipping Anthropic integration: {}",
                    exc,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to initialize Anthropic client: {}",
                    exc,
                )
        except ImportError:
            logger.warning(
                "Anthropic provider configuration requested but AnthropicConfig unavailable",
            )

    return config, client, async_client


def _configure_gemini(inputs: ProviderSetupInputs) -> tuple[Any | None, Any | None]:
    """Configure Google Gemini clients when requested."""

    config = None
    client = None

    if inputs.gemini_config or inputs.gemini_api_key or inputs.gemini_model:
        try:
            from .providers import GeminiConfig

            config = inputs.gemini_config or GeminiConfig(
                api_key=inputs.gemini_api_key,
                model=inputs.gemini_model,
            )

            if inputs.gemini_model and getattr(config, "model", None) in {None, ""}:
                config.model = inputs.gemini_model
            if inputs.gemini_api_key and getattr(config, "api_key", None) in {None, ""}:
                config.api_key = inputs.gemini_api_key

            try:
                client = config.create_client()
                logger.info("Google Gemini provider configured successfully")
            except ImportError as exc:
                logger.warning(
                    "Google Generative AI SDK is not installed; skipping Gemini integration: {}",
                    exc,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to initialize Google Gemini client: {}",
                    exc,
                )
        except ImportError:
            logger.warning(
                "Gemini provider configuration requested but GeminiConfig unavailable",
            )

    return config, client


def _initialize_agents(
    *,
    provider: Any | None,
    api_key: str,
    model: str,
    inputs: ProviderSetupInputs,
    provider_registry: ProviderRegistry,
    task_routes: Mapping[str, TaskRouteSpec],
) -> tuple[Any | None, Any | None, Any | None, bool, bool]:
    """Create memory/search agents and the conscious agent when enabled."""

    memory_agent = None
    search_engine = None
    conscious_agent = None

    conscious_ingest = inputs.conscious_ingest
    auto_ingest = inputs.auto_ingest

    if inputs.sovereign_ingest:
        logger.info("Sovereign ingest enabled; skipping LLM agent initialization.")
    else:
        try:
            from ..agents.memory_agent import MemoryAgent
            from ..agents.retrieval_agent import MemorySearchEngine

            agent_kwargs = {
                "model": model,
                "provider_registry": provider_registry,
                "task_routes": dict(task_routes),
            }
            if provider is not None:
                agent_kwargs["provider_config"] = provider
            else:
                agent_kwargs["api_key"] = api_key

            memory_agent = MemoryAgent(**agent_kwargs)
            search_engine = MemorySearchEngine(**agent_kwargs)
            logger.info("Agents initialized successfully with model: {}", model)
        except ImportError as exc:
            logger.warning(
                "Failed to import LLM agents: {}. Memory ingestion disabled.",
                exc,
            )
            conscious_ingest = False
            auto_ingest = False
        except Exception as exc:
            logger.warning(
                "Failed to initialize LLM agents: {}. Memory ingestion disabled.",
                exc,
            )
            conscious_ingest = False
            auto_ingest = False
            memory_agent = None
            search_engine = None

    if (conscious_ingest or auto_ingest) and inputs.enable_short_term:
        try:
            from ..agents.conscious_agent import ConsciousAgent

            conscious_agent = ConsciousAgent(
                use_heuristics=inputs.use_lightweight_conscious_ingest
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize conscious agent: {}. Disabling conscious ingest.",
                exc,
            )
            conscious_agent = None
            conscious_ingest = False
            auto_ingest = False
    else:
        conscious_ingest = False
        auto_ingest = False

    return (
        memory_agent,
        search_engine,
        conscious_agent,
        conscious_ingest,
        auto_ingest,
    )
