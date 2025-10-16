"""
Configuration manager for Memoria
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger

from ..policy.schemas import merge_policy_sections
from ..utils.exceptions import ConfigurationError
from .settings import MemoriaSettings, TeamMode


class ConfigManager:
    """Central configuration manager for Memoria"""

    _instance: ConfigManager | None = None
    _settings: MemoriaSettings | None = None

    def __new__(cls) -> ConfigManager:
        """Singleton pattern for configuration manager"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration manager"""
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._config_sources = []
            self._loaded_config_path: str | None = None
            self._env_overrides: list[str] = []
            self._load_default_config()

    def _load_default_config(self) -> None:
        """Load default configuration"""
        try:
            self._settings = MemoriaSettings()
            self._config_sources.append("defaults")
            self._env_overrides = []
            logger.debug("Loaded default configuration")
            self._post_load_actions()

        except Exception as e:
            raise ConfigurationError(f"Failed to load default configuration: {e}")

    def load_from_env(self) -> None:
        """Load configuration from environment variables"""
        try:
            settings, used_keys = MemoriaSettings.from_env_with_metadata()
            self._settings = settings
            if used_keys:
                if "environment" not in self._config_sources:
                    self._config_sources.append("environment")
                self._env_overrides = sorted(used_keys)
                logger.info(
                    "Configuration loaded from environment variables: {}",
                    ", ".join(self._env_overrides),
                )
            else:
                self._env_overrides = []
                logger.info(
                    "Environment load requested but no MEMORIA_* variables were set",
                )
            self._post_load_actions()

        except Exception as e:
            logger.warning(f"Failed to load configuration from environment: {e}")
            raise ConfigurationError(f"Environment configuration error: {e}")

    def load_from_file(self, config_path: str | Path) -> None:
        """Load configuration from file"""
        try:
            config_path = Path(config_path)
            self._settings = MemoriaSettings.from_file(config_path)
            self._config_sources.append(str(config_path))
            self._loaded_config_path = str(config_path)
            logger.info(f"Configuration loaded from file: {config_path}")
            self._post_load_actions()

        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {config_path}")
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration from file {config_path}: {e}")
            raise ConfigurationError(f"File configuration error: {e}")

    def auto_load(self) -> None:
        """Automatically load configuration from multiple sources in priority order"""
        self._env_overrides = []
        config_locations = [
            # Environment-specific paths
            os.getenv("MEMORIA_CONFIG_PATH"),
            "memoria.json",
            "memoria.yaml",
            "memoria.yml",
            "config/memoria.json",
            "config/memoria.yaml",
            "config/memoria.yml",
            Path.home() / ".memoria" / "config.json",
            Path.home() / ".memoria" / "config.yaml",
            "/etc/memoria/config.json",
            "/etc/memoria/config.yaml",
        ]

        # Try to load from the first available config file
        for config_path in config_locations:
            if config_path and Path(config_path).exists():
                try:
                    self.load_from_file(config_path)
                    break
                except ConfigurationError:
                    continue

        # Override with environment variables if present
        try:
            env_settings, used_keys = MemoriaSettings.from_env_with_metadata()
        except Exception as e:
            logger.warning(f"Failed to parse environment configuration: {e}")
            used_keys = set()
            env_settings = None

        if env_settings and used_keys:
            try:
                if self._settings:
                    # Merge environment settings with existing settings
                    self._merge_settings(env_settings)
                else:
                    self._settings = env_settings

                if "environment" not in self._config_sources:
                    self._config_sources.append("environment")

                self._env_overrides = sorted(used_keys)
                logger.info(
                    "Environment variables merged into configuration: {}",
                    ", ".join(self._env_overrides),
                )
            except Exception as e:
                logger.warning(f"Failed to merge environment configuration: {e}")

        self._post_load_actions()

    def _merge_settings(self, new_settings: MemoriaSettings) -> None:
        """Merge new settings with existing settings"""
        if self._settings is None:
            self._settings = new_settings
            return

        # Convert to dict, merge, and recreate
        current_dict = self._settings.dict()
        new_dict = new_settings.dict()

        # Deep merge dictionaries
        merged_dict = self._deep_merge_dicts(current_dict, new_dict)
        self._settings = MemoriaSettings(**merged_dict)

    def _deep_merge_dicts(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()

        for key, value in override.items():
            if key == "policy":
                result[key] = merge_policy_sections(result.get(key), value)
            elif (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value

        return result

    def _post_load_actions(self) -> None:
        """Run post-load validations and setup"""
        if self._settings is None:
            return
        if not self._settings.use_db_clusters and self._settings.cluster_index_path:
            self._verify_cluster_index_path()

    def _verify_cluster_index_path(self) -> None:
        """Ensure cluster index directory exists and is writable"""
        if not self._settings.cluster_index_path:
            return
        path = Path(self._settings.cluster_index_path)
        directory = path.parent

        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create cluster index directory {directory}: {e}")
            raise ConfigurationError(
                f"Cannot create cluster index directory: {directory}"
            ) from e

        if not os.access(directory, os.W_OK):
            logger.error(f"Cluster index directory is not writable: {directory}")
            raise ConfigurationError(
                f"Cluster index directory is not writable: {directory}"
            )

    def save_to_file(self, config_path: str | Path, format: str = "json") -> None:
        """Save current configuration to file"""
        if self._settings is None:
            raise ConfigurationError("No configuration loaded to save")

        try:
            self._settings.to_file(config_path, format)
            logger.info(f"Configuration saved to: {config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise ConfigurationError(f"Failed to save configuration: {e}")

    def save_configuration(self) -> None:
        """Save the current configuration back to its original file."""
        if self._settings is None:
            raise ConfigurationError("No configuration loaded to save")

        save_path_str = self._loaded_config_path or "config.yaml"
        save_path = Path(save_path_str)
        format = "yaml" if save_path.suffix.lower() in (".yaml", ".yml") else "json"

        try:
            self.save_to_file(save_path, format=format)
        except ConfigurationError as e:
            logger.error(
                f"Failed to automatically save configuration to {save_path}: {e}"
            )
            # Optionally re-raise or handle as a non-fatal error
            raise

    def get_settings(self) -> MemoriaSettings:
        """Get current settings"""
        if self._settings is None:
            raise ConfigurationError("Configuration not loaded")
        return self._settings

    def export_settings(self, *, include_sensitive: bool = False) -> dict[str, Any]:
        """Return a serialisable snapshot of the current settings."""

        if self._settings is None:
            raise ConfigurationError("Configuration not loaded")
        return self._settings.export(include_sensitive=include_sensitive)

    def update_setting(self, key_path: str, value: Any) -> None:
        """Update a specific setting using dot notation (e.g., 'database.pool_size')"""
        if self._settings is None:
            raise ConfigurationError("Configuration not loaded")

        try:
            # Convert to dict for easier manipulation
            settings_dict = self._settings.dict()

            # Navigate to the nested key
            keys = key_path.split(".")
            current = settings_dict

            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Set the value
            current[keys[-1]] = value

            # Recreate settings object
            self._settings = MemoriaSettings(**settings_dict)
            masked_value = "***" if key_path == "database.connection_string" else value
            logger.debug(f"Updated setting {key_path} = {masked_value}")

        except Exception as e:
            logger.error(f"Failed to update setting {key_path}: {e}")
            raise ConfigurationError(f"Setting update error: {e}")

    def get_setting(self, key_path: str, default: Any = None) -> Any:
        """Get a specific setting using dot notation"""
        if self._settings is None:
            raise ConfigurationError("Configuration not loaded")

        try:
            # Navigate through the settings
            current = self._settings.dict()
            keys = key_path.split(".")

            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default

            return current

        except Exception:
            return default

    def validate_configuration(self) -> bool:
        """Validate current configuration"""
        if self._settings is None:
            return False

        try:
            # Pydantic validation happens automatically during object creation
            # Additional custom validation can be added here

            # Validate database connection if possible
            db_url = self._settings.get_database_url()
            if not db_url:
                logger.error("Database connection string is required")
                return False

            # Validate API keys if conscious ingestion is enabled
            if (
                self._settings.agents.conscious_ingest
                and not self._settings.agents.openai_api_key
            ):
                logger.warning("OpenAI API key is required for conscious ingestion")

            logger.info("Configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def get_config_info(self) -> dict[str, Any]:
        """Get information about current configuration"""
        team_mode_value: str | None = None
        default_team_id: str | None = None
        if self._settings is not None:
            memory_cfg = getattr(self._settings, "memory", None)
            if memory_cfg is not None:
                raw_mode = getattr(memory_cfg, "team_mode", TeamMode.DISABLED)
                if isinstance(raw_mode, TeamMode):
                    team_mode_value = raw_mode.value
                elif raw_mode is not None:
                    team_mode_value = str(raw_mode)
                default_team_id = getattr(memory_cfg, "team_default_id", None)
        return {
            "loaded": self._settings is not None,
            "sources": self._config_sources.copy(),
            "version": self._settings.version if self._settings else None,
            "debug_mode": self._settings.debug if self._settings else False,
            "is_production": (
                self._settings.is_production() if self._settings else False
            ),
            "env_overrides": self._env_overrides.copy(),
            "team_mode": team_mode_value,
            "default_team_id": default_team_id,
        }

    def get_team_configuration(self) -> dict[str, Any]:
        """Return the active team-related configuration values."""

        if self._settings is None:
            raise ConfigurationError("Configuration not loaded")

        memory_cfg = getattr(self._settings, "memory", None)
        if memory_cfg is None:
            return {
                "mode": TeamMode.DISABLED.value,
                "default_team_id": None,
                "share_by_default": False,
                "enforce_membership": True,
            }

        raw_mode = getattr(memory_cfg, "team_mode", TeamMode.DISABLED)
        if isinstance(raw_mode, TeamMode):
            mode_value = raw_mode.value
        else:
            mode_value = str(raw_mode)

        return {
            "mode": mode_value,
            "default_team_id": getattr(memory_cfg, "team_default_id", None),
            "share_by_default": bool(
                getattr(memory_cfg, "team_share_by_default", False)
            ),
            "enforce_membership": bool(
                getattr(memory_cfg, "team_enforce_membership", True)
            ),
        }

    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults"""
        self._settings = MemoriaSettings()
        self._config_sources = ["defaults"]
        self._env_overrides = []
        logger.info("Configuration reset to defaults")
        self._post_load_actions()

    def setup_logging(self) -> None:
        """Setup logging based on current configuration"""
        if self._settings is None:
            raise ConfigurationError("Configuration not loaded")

        try:
            # Import here to avoid circular import
            from ..utils.logging import LoggingManager

            LoggingManager.setup_logging(
                self._settings.logging, verbose=self._settings.verbose
            )

            if self._settings.verbose:
                logger.info("Verbose logging enabled through ConfigManager")
        except Exception as e:
            logger.error(f"Failed to setup logging: {e}")
            raise ConfigurationError(f"Logging setup error: {e}")

    @classmethod
    def get_instance(cls) -> ConfigManager:
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
