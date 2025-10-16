"""Plugin system primitives for Memoria."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import import_module
from inspect import isclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from memoria.config.settings import PluginSettings
    from memoria.core.memory import Memoria


@runtime_checkable
class BasePlugin(Protocol):
    """Runtime protocol implemented by Memoria plugins."""

    def initialize(self, memoria: Memoria) -> None:
        """Bind the plugin to the Memoria instance."""

    def on_memoria_ready(
        self, memoria: Memoria
    ) -> None:  # pragma: no cover - optional hook
        """Optional hook invoked once the Memoria instance is fully initialized."""

    def shutdown(self) -> None:  # pragma: no cover - optional hook
        """Optional hook invoked when Memoria is shutting down."""


@dataclass(slots=True)
class LoadedPlugin:
    """Container describing a loaded plugin instance."""

    name: str
    import_path: str
    instance: BasePlugin
    options: dict[str, Any]


class PluginRegistry:
    """Track successfully loaded plugins and load failures."""

    def __init__(self) -> None:
        self._plugins: list[LoadedPlugin] = []
        self._by_name: dict[str, LoadedPlugin] = {}
        self._failures: dict[str, str] = {}
        self._disabled: list[str] = []

    def register(self, plugin: LoadedPlugin) -> None:
        """Register a successfully loaded plugin."""

        logger.debug(
            "Registered plugin '{}' from '{}'", plugin.name, plugin.import_path
        )
        self._plugins.append(plugin)
        self._by_name[plugin.name] = plugin

    def mark_disabled(self, name: str) -> None:
        """Record a plugin that was explicitly disabled."""

        logger.debug("Plugin '{}' disabled via configuration", name)
        self._disabled.append(name)

    def mark_failure(self, name: str, reason: str) -> None:
        """Record a plugin that failed to load."""

        logger.warning("Plugin '{}' failed to load: {}", name, reason)
        self._failures[name] = reason

    def get(self, name: str) -> BasePlugin | None:
        """Return the plugin instance registered under ``name`` if present."""

        entry = self._by_name.get(name)
        return entry.instance if entry else None

    def iter_plugins(self) -> Iterable[BasePlugin]:
        """Iterate over registered plugin instances."""

        for entry in self._plugins:
            yield entry.instance

    @property
    def failures(self) -> dict[str, str]:
        """Return a mapping of plugin name to failure reason."""

        return dict(self._failures)

    @property
    def disabled(self) -> list[str]:
        """Return the list of plugins that were disabled."""

        return list(self._disabled)

    @property
    def loaded(self) -> list[LoadedPlugin]:
        """Return information about all loaded plugins."""

        return list(self._plugins)

    def shutdown_all(self) -> None:
        """Invoke the shutdown hook on all registered plugins."""

        for entry in self._plugins:
            shutdown = getattr(entry.instance, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.warning(
                        "Plugin '{}' failed during shutdown: {}", entry.name, exc
                    )


def _extract_config_value(config: Any, key: str, default: Any = None) -> Any:
    if hasattr(config, key):
        return getattr(config, key)
    if isinstance(config, dict):
        return config.get(key, default)
    return default


def _import_plugin(import_path: str) -> Any:
    module_path: str
    attribute: str | None
    if ":" in import_path:
        module_path, attribute = import_path.split(":", 1)
    else:
        parts = import_path.rsplit(".", 1)
        if len(parts) == 2:
            module_path, attribute = parts
        else:
            module_path = import_path
            attribute = None

    module = import_module(module_path)
    if attribute:
        try:
            return getattr(module, attribute)
        except AttributeError as exc:  # pragma: no cover - defensive guard
            raise ImportError(
                f"Plugin attribute '{attribute}' not found in '{module_path}'"
            ) from exc
    return module


def _instantiate_plugin(target: Any, options: dict[str, Any]) -> BasePlugin:
    if isclass(target):
        instance = target(**options) if options else target()
    elif isinstance(target, BasePlugin):  # type: ignore[arg-type]
        instance = target
    elif callable(target):
        instance = target(**options) if options else target()
    else:
        instance = target

    if not isinstance(instance, BasePlugin):
        required_methods = ("initialize",)
        missing = [
            name
            for name in required_methods
            if not callable(getattr(instance, name, None))
        ]
        if missing:
            raise TypeError(
                "Plugin object missing required methods: {}".format(", ".join(missing))
            )
        raise TypeError("Loaded object does not implement the BasePlugin protocol")

    return instance


def _resolve_plugin_name(
    configured_name: str | None, plugin_object: Any, import_path: str
) -> str:
    if configured_name:
        return configured_name
    candidate = getattr(plugin_object, "name", None)
    if isinstance(candidate, str) and candidate.strip():
        return candidate
    if isclass(plugin_object):
        return plugin_object.__name__
    return import_path


def load_plugins(
    memoria: Memoria,
    plugin_settings: Sequence[PluginSettings] | None,
    *,
    registry: PluginRegistry | None = None,
) -> PluginRegistry:
    """Instantiate and initialize plugins declared in configuration."""

    registry = registry or PluginRegistry()
    if not plugin_settings:
        return registry

    for config in plugin_settings:
        import_path = _extract_config_value(config, "import_path")
        configured_name = _extract_config_value(config, "name")
        enabled = _extract_config_value(config, "enabled", True)
        options = _extract_config_value(config, "options", {}) or {}

        display_name = str(configured_name or import_path or "<unnamed>")

        if not enabled:
            registry.mark_disabled(display_name)
            continue

        if not import_path or not str(import_path).strip():
            registry.mark_failure(display_name, "Missing import path")
            continue

        try:
            target = _import_plugin(str(import_path))
            plugin_instance = _instantiate_plugin(target, dict(options))
            plugin_name = _resolve_plugin_name(
                configured_name, plugin_instance, str(import_path)
            )

            initialize = getattr(plugin_instance, "initialize", None)
            if not callable(initialize):
                raise TypeError(
                    f"Plugin '{plugin_name}' does not provide an 'initialize' method"
                )

            initialize(memoria)

            ready_hook = getattr(plugin_instance, "on_memoria_ready", None)
            if callable(ready_hook):
                ready_hook(memoria)

            registry.register(
                LoadedPlugin(
                    name=plugin_name,
                    import_path=str(import_path),
                    instance=plugin_instance,
                    options=dict(options),
                )
            )
        except Exception as exc:  # pragma: no cover - exceptions exercised in tests
            registry.mark_failure(display_name, str(exc))

    return registry


__all__ = [
    "BasePlugin",
    "LoadedPlugin",
    "PluginRegistry",
    "load_plugins",
]
