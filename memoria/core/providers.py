"""Provider configuration and routing utilities for LLM providers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


class ProviderType(Enum):
    """Supported LLM provider types"""

    OPENAI = "openai"
    AZURE = "azure"
    CUSTOM = "custom"
    OPENAI_COMPATIBLE = "openai_compatible"  # For OpenAI-compatible APIs
    ANTHROPIC = "anthropic"
    GOOGLE_GEMINI = "google_gemini"


class ProviderUnavailableError(RuntimeError):
    """Raised when a provider client cannot be created or is unavailable."""


@dataclass(slots=True)
class ProviderRouteMetadata:
    """Metadata describing routing preferences for a provider."""

    preferred_tasks: tuple[str, ...] = ("any",)
    cost_profile: str = "standard"
    supports_structured_outputs: bool = True


@dataclass(slots=True)
class TaskRouteSpec:
    """Configuration describing how a task should be routed to a provider."""

    provider: str
    model: str | None = None
    fallback: tuple[str, ...] = ()


@dataclass(slots=True)
class RegisteredProvider:
    """Runtime record for a configured provider client."""

    name: str
    provider_type: ProviderType
    model: str | None = None
    metadata: ProviderRouteMetadata = field(default_factory=ProviderRouteMetadata)
    client_factory: Callable[[], Any] | None = None
    async_client_factory: Callable[[], Any] | None = None
    client_instance: Any | None = None
    async_client_instance: Any | None = None
    available: bool = True

    def is_available(self) -> bool:
        """Return ``True`` when the provider is available for selection."""

        return bool(self.available)

    def mark_unavailable(self) -> None:
        """Mark this provider as unavailable for future selections."""

        self.available = False

    def get_client(self) -> Any:
        """Return or instantiate the synchronous client."""

        if not self.available:
            raise ProviderUnavailableError(f"Provider '{self.name}' is unavailable")

        if self.client_instance is not None:
            return self.client_instance

        if self.client_factory is None:
            raise ProviderUnavailableError(
                f"Provider '{self.name}' does not define a synchronous client"
            )

        try:
            self.client_instance = self.client_factory()
        except Exception as exc:  # pragma: no cover - defensive guard
            self.available = False
            raise ProviderUnavailableError(str(exc)) from exc
        return self.client_instance

    def get_async_client(self) -> Any:
        """Return or instantiate the asynchronous client."""

        if not self.available:
            raise ProviderUnavailableError(f"Provider '{self.name}' is unavailable")

        if self.async_client_instance is not None:
            return self.async_client_instance

        if self.async_client_factory is None:
            raise ProviderUnavailableError(
                f"Provider '{self.name}' does not define an async client"
            )

        try:
            self.async_client_instance = self.async_client_factory()
        except Exception as exc:  # pragma: no cover - defensive guard
            self.available = False
            raise ProviderUnavailableError(str(exc)) from exc
        return self.async_client_instance


@dataclass(slots=True)
class ProviderSelection:
    """Result of routing a task to a specific provider."""

    task: str
    provider: RegisteredProvider
    model: str | None
    fallback_used: bool
    preferred: bool

    @property
    def provider_name(self) -> str:
        return self.provider.name

    @property
    def provider_type(self) -> ProviderType:
        return self.provider.provider_type

    def get_client(self) -> Any:
        """Return a synchronous client for the selection."""

        return self.provider.get_client()

    def get_async_client(self) -> Any:
        """Return an asynchronous client for the selection."""

        return self.provider.get_async_client()


class ProviderRegistry:
    """Registry of configured providers with routing utilities."""

    _COST_ORDER = {"low": 0, "standard": 1, "medium": 1, "high": 2, "premium": 3}

    def __init__(self) -> None:
        self._providers: dict[str, RegisteredProvider] = {}
        self._aliases: dict[str, str] = {}
        self._task_routes: dict[str, TaskRouteSpec] = {}
        self._primary_provider: str | None = None

    # Public API -----------------------------------------------------

    def register(
        self,
        provider: RegisteredProvider,
        *,
        aliases: Iterable[str] | None = None,
        is_primary: bool = False,
    ) -> None:
        """Register a provider and optional aliases."""

        name = provider.name
        if name in self._providers:
            logger.debug("Updating existing provider registration: {}", name)
        self._providers[name] = provider

        if aliases:
            for alias in aliases:
                normalized = alias.strip().lower()
                if not normalized:
                    continue
                self._aliases[normalized] = name

        if is_primary or self._primary_provider is None:
            self._primary_provider = name

    def set_task_routes(self, routes: Mapping[str, TaskRouteSpec]) -> None:
        """Replace task routing configuration."""

        self._task_routes = dict(routes)

    def get_task_routes(self) -> Mapping[str, TaskRouteSpec]:
        """Return a snapshot of configured task routes."""

        return dict(self._task_routes)

    def providers(self) -> Sequence[RegisteredProvider]:
        """Return registered providers."""

        return list(self._providers.values())

    def select(
        self,
        task: str,
        *,
        require_structured: bool = False,
        preferred_provider: str | None = None,
        exclude_providers: Iterable[str] | None = None,
    ) -> ProviderSelection | None:
        """Select an appropriate provider for ``task``."""

        exclude = {name.lower() for name in (exclude_providers or [])}
        candidate_queue: list[tuple[str, str | None, str]] = []

        if preferred_provider:
            candidate_queue.append(
                (*self._parse_identifier(preferred_provider), "preferred")
            )

        route = self._task_routes.get(task)
        if route:
            candidate_queue.append((route.provider, route.model, "primary"))
            for candidate in route.fallback:
                candidate_queue.append((*self._parse_identifier(candidate), "fallback"))
        else:
            candidate_queue.append(("primary", None, "primary"))

        seen: set[tuple[str, str | None]] = set()
        for provider_name, model_override, origin in candidate_queue:
            resolved = self._resolve_provider_name(provider_name)
            key = (resolved, model_override)
            if not resolved or key in seen:
                continue
            seen.add(key)
            if resolved.lower() in exclude:
                continue
            provider = self._providers.get(resolved)
            if provider is None or not provider.is_available():
                continue
            if require_structured and not provider.metadata.supports_structured_outputs:
                continue
            model = model_override
            if model is None and route:
                model = route.model
            if model is None:
                model = provider.model
            return ProviderSelection(
                task=task,
                provider=provider,
                model=model,
                fallback_used=origin == "fallback",
                preferred=origin == "preferred",
            )

        # Fallback to cost-based ordering for providers supporting the task
        ranked = self._rank_providers_for_task(
            task, require_structured=require_structured
        )
        for provider in ranked:
            if provider.name.lower() in exclude:
                continue
            return ProviderSelection(
                task=task,
                provider=provider,
                model=provider.model,
                fallback_used=True,
                preferred=False,
            )

        return None

    def mark_unavailable(self, provider_name: str) -> None:
        """Mark ``provider_name`` as unavailable."""

        resolved = self._resolve_provider_name(provider_name)
        provider = self._providers.get(resolved)
        if provider is not None:
            provider.mark_unavailable()

    # Internal helpers ------------------------------------------------

    def _parse_identifier(self, identifier: str) -> tuple[str, str | None]:
        value = (identifier or "").strip()
        if not value:
            return "", None
        if ":" in value:
            provider_name, model = value.split(":", 1)
            return provider_name.strip(), model.strip() or None
        return value, None

    def _resolve_provider_name(self, name: str | None) -> str:
        if not name:
            return self._primary_provider or ""
        candidate = name.strip().lower()
        if candidate == "primary":
            return self._primary_provider or ""
        if candidate in self._aliases:
            return self._aliases[candidate]
        if candidate in self._providers:
            return candidate
        return candidate

    def _rank_providers_for_task(
        self, task: str, *, require_structured: bool
    ) -> list[RegisteredProvider]:
        def _score(provider: RegisteredProvider) -> tuple[int, int]:
            cost = self._COST_ORDER.get(
                provider.metadata.cost_profile.strip().lower(), 1
            )
            preferred = provider.metadata.preferred_tasks
            specificity = 1
            if task in preferred:
                specificity = 0
            elif "any" in preferred:
                specificity = 1
            else:
                specificity = 2
            return (specificity, cost)

        eligible = [
            provider
            for provider in self._providers.values()
            if provider.is_available()
            and (
                not require_structured or provider.metadata.supports_structured_outputs
            )
        ]

        eligible.sort(key=_score)
        return eligible


@dataclass
class ProviderConfig:
    """
    Configuration for LLM providers with support for OpenAI, Azure, and custom endpoints.

    This class provides a unified interface for configuring different LLM providers
    while maintaining backward compatibility with existing OpenAI-only configuration.
    """

    # Common parameters
    api_key: str | None = None
    api_type: str | None = None  # "openai", "azure", or custom
    base_url: str | None = None  # Custom endpoint URL
    timeout: float | None = None
    max_retries: int | None = None

    # Azure-specific parameters
    azure_endpoint: str | None = None
    azure_deployment: str | None = None
    api_version: str | None = None
    azure_ad_token: str | None = None

    # OpenAI-specific parameters
    organization: str | None = None
    project: str | None = None

    # Model configuration
    model: str | None = (
        None  # User can specify model, defaults to gpt-4o-mini if not set
    )

    # Additional headers for custom providers
    default_headers: dict[str, str] | None = None
    default_query: dict[str, Any] | None = None
    extra_params: dict[str, Any] | None = None

    # HTTP client configuration
    http_client: Any | None = None

    @classmethod
    def from_openai(
        cls, api_key: str | None = None, model: str | None = None, **kwargs
    ):
        """Create configuration for standard OpenAI"""
        return cls(api_key=api_key, api_type="openai", model=model, **kwargs)

    @classmethod
    def from_azure(
        cls,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
        model: str | None = None,
        **kwargs,
    ):
        """Create configuration for Azure OpenAI"""
        return cls(
            api_key=api_key,
            api_type="azure",
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            model=model,
            **kwargs,
        )

    @classmethod
    def from_custom(
        cls,
        base_url: str,
        api_key: str | None = None,
        model: str | None = None,
        **kwargs,
    ):
        """Create configuration for custom OpenAI-compatible endpoints"""
        return cls(
            api_key=api_key, api_type="custom", base_url=base_url, model=model, **kwargs
        )

    def get_openai_client_kwargs(self) -> dict[str, Any]:
        """
        Get kwargs for OpenAI client initialization based on provider type.

        Returns:
            Dictionary of parameters to pass to OpenAI client constructor
        """
        kwargs = {}

        # Always include API key if provided
        if self.api_key:
            kwargs["api_key"] = self.api_key

        if self.api_type == "azure":
            # Azure OpenAI configuration
            if self.azure_endpoint:
                kwargs["azure_endpoint"] = self.azure_endpoint
            if self.azure_deployment:
                kwargs["azure_deployment"] = self.azure_deployment
            if self.api_version:
                kwargs["api_version"] = self.api_version
            if self.azure_ad_token:
                kwargs["azure_ad_token"] = self.azure_ad_token
            # For Azure, we need to use AzureOpenAI client
            kwargs["_use_azure_client"] = True

        elif self.api_type == "custom" or self.api_type == "openai_compatible":
            # Custom endpoint configuration
            if self.base_url:
                kwargs["base_url"] = self.base_url

        elif self.api_type == "openai":
            # Standard OpenAI configuration
            if self.organization:
                kwargs["organization"] = self.organization
            if self.project:
                kwargs["project"] = self.project

        # Common parameters
        if self.timeout:
            kwargs["timeout"] = self.timeout
        if self.max_retries:
            kwargs["max_retries"] = self.max_retries
        if self.default_headers:
            kwargs["default_headers"] = self.default_headers
        if self.default_query:
            kwargs["default_query"] = self.default_query
        if self.http_client:
            kwargs["http_client"] = self.http_client

        return kwargs

    def create_client(self):
        """
        Create the appropriate OpenAI client based on configuration.

        Returns:
            OpenAI or AzureOpenAI client instance
        """
        import openai

        kwargs = self.get_openai_client_kwargs()

        # Check if we should use Azure client
        if kwargs.pop("_use_azure_client", False):
            # Use Azure OpenAI client
            from openai import AzureOpenAI

            return AzureOpenAI(**kwargs)
        else:
            # Use standard OpenAI client (works for OpenAI and custom endpoints)
            return openai.OpenAI(**kwargs)

    def create_async_client(self):
        """
        Create the appropriate async OpenAI client based on configuration.

        Returns:
            AsyncOpenAI or AsyncAzureOpenAI client instance
        """
        import openai

        kwargs = self.get_openai_client_kwargs()

        # Check if we should use Azure client
        if kwargs.pop("_use_azure_client", False):
            # Use Azure OpenAI async client
            from openai import AsyncAzureOpenAI

            return AsyncAzureOpenAI(**kwargs)
        else:
            # Use standard async OpenAI client
            return openai.AsyncOpenAI(**kwargs)


@dataclass
class AnthropicConfig:
    """Configuration helper for Anthropic Claude models."""

    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    timeout: float | None = None
    default_headers: dict[str, str] | None = None

    def _build_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by the Anthropic SDK constructors."""

        kwargs: dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout:
            kwargs["timeout"] = self.timeout
        if self.default_headers:
            kwargs["default_headers"] = self.default_headers
        return kwargs

    def create_client(self) -> Any:
        """Instantiate the synchronous Anthropic client."""

        import anthropic

        kwargs = self._build_kwargs()
        return anthropic.Anthropic(**kwargs)

    def create_async_client(self) -> Any:
        """Instantiate the asynchronous Anthropic client."""

        import anthropic

        kwargs = self._build_kwargs()
        return anthropic.AsyncAnthropic(**kwargs)


@dataclass
class GeminiConfig:
    """Configuration helper for Google Gemini models."""

    api_key: str | None = None
    model: str | None = None
    client_options: dict[str, Any] | None = None
    generation_config: dict[str, Any] | None = None
    safety_settings: Any | None = None

    def _configure_sdk(self) -> Any:
        """Configure the underlying Google Generative AI SDK."""

        import google.generativeai as genai

        kwargs: dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.client_options:
            kwargs["client_options"] = self.client_options
        if kwargs:
            genai.configure(**kwargs)
        return genai

    def create_client(self) -> Any:
        """Instantiate a GenerativeModel client for Gemini."""

        genai = self._configure_sdk()
        model_name = self.model or "gemini-1.5-pro"
        kwargs: dict[str, Any] = {}
        if self.generation_config:
            kwargs["generation_config"] = self.generation_config
        if self.safety_settings is not None:
            kwargs["safety_settings"] = self.safety_settings
        return genai.GenerativeModel(model_name, **kwargs)

    def create_async_client(self) -> Any:
        """Return an async-compatible Gemini client.

        The Google Generative AI SDK currently exposes a single client type. We
        therefore return the synchronous client instance for async usage to
        maintain a consistent interface with other provider configurations.
        """

        return self.create_client()


def detect_provider_from_env() -> ProviderConfig:
    """
    Create provider configuration from environment variables WITHOUT automatic detection.

    This function ONLY uses standard OpenAI configuration by default.
    It does NOT automatically detect or prioritize Azure or custom providers.

    Only use specific providers if explicitly configured via Memoria constructor parameters.

    Returns:
        Standard OpenAI ProviderConfig instance (never auto-detects other providers)
    """
    import os

    # Get model from environment (optional, defaults to gpt-4o-mini if not set)
    model = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini"

    # ALWAYS default to standard OpenAI - no automatic detection
    logger.info("Provider configuration: Using standard OpenAI (no auto-detection)")
    return ProviderConfig.from_openai(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORGANIZATION"),
        project=os.getenv("OPENAI_PROJECT"),
        model=model,
    )
