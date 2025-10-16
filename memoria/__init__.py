"""
Memoria - The Open-Source Memory Layer for AI Agents & Multi-Agent Systems v0.9.0a0 (Alpha)

Professional-grade memory layer with comprehensive error handling, configuration
management, and modular architecture for production AI systems.
"""

__version__ = "0.9.0a0"
__author__ = "Adrian Hau"
__email__ = "maintainers@memoria.dev"

from typing import Any

# Configuration system
from .config import (
    AgentSettings,
    ConfigManager,
    DatabaseSettings,
    LoggingSettings,
    MemoriaSettings,
    SyncSettings,
)
from .core.database import DatabaseManager

# Core components
from .core.memory import Memoria

# Database system
from .database.connectors import MySQLConnector, PostgreSQLConnector, SQLiteConnector
from .database.queries import BaseQueries, ChatQueries, EntityQueries, MemoryQueries

# Wrapper integrations
from .integrations import MemoriaAnthropic, MemoriaOpenAI
from .plugins import BasePlugin, PluginRegistry, load_plugins

# Tools and integrations
from .tools.memory_tool import MemoryTool, create_memory_search_tool, create_memory_tool

# Utils and models
from .utils import (  # Pydantic models; Enhanced exceptions; Validators and helpers; Logging
    AgentError,
    AsyncUtils,
    AuthenticationError,
    ConfigurationError,
    ConversationContext,
    DatabaseError,
    DataValidator,
    DateTimeUtils,
    EntityType,
    ExceptionHandler,
    ExtractedEntities,
    FileUtils,
    IntegrationError,
    JsonUtils,
    LoggingManager,
    MemoriaError,
    MemoryCategory,
    MemoryCategoryType,
    MemoryImportance,
    MemoryNotFoundError,
    MemoryValidator,
    PerformanceUtils,
    ProcessedMemory,
    ProcessingError,
    RateLimitError,
    ResourceExhaustedError,
    RetentionType,
    RetryUtils,
    StringUtils,
    TimeoutError,
    ValidationError,
    get_logger,
)

# Memory agents (dynamically imported to avoid import errors)
MemoryAgent: Any | None = None
MemorySearchEngine: Any | None = None
_AGENTS_AVAILABLE = False

try:
    from .agents.memory_agent import MemoryAgent
    from .agents.retrieval_agent import MemorySearchEngine

    _AGENTS_AVAILABLE = True
except ImportError:
    # Agents are not available, use placeholder None values
    pass

# Build __all__ list dynamically based on available components
_all_components = [
    # Core
    "Memoria",
    "DatabaseManager",
    # Configuration
    "MemoriaSettings",
    "DatabaseSettings",
    "AgentSettings",
    "LoggingSettings",
    "SyncSettings",
    "ConfigManager",
    # Database
    "SQLiteConnector",
    "PostgreSQLConnector",
    "MySQLConnector",
    "BaseQueries",
    "MemoryQueries",
    "ChatQueries",
    "EntityQueries",
    # Tools
    "MemoryTool",
    "create_memory_tool",
    "create_memory_search_tool",
    # Integrations
    "MemoriaOpenAI",
    "MemoriaAnthropic",
    # Plugins
    "BasePlugin",
    "PluginRegistry",
    "load_plugins",
    # Pydantic Models
    "ProcessedMemory",
    "MemoryCategory",
    "ExtractedEntities",
    "MemoryImportance",
    "ConversationContext",
    "MemoryCategoryType",
    "RetentionType",
    "EntityType",
    # Enhanced Exceptions
    "MemoriaError",
    "DatabaseError",
    "AgentError",
    "ConfigurationError",
    "ValidationError",
    "IntegrationError",
    "AuthenticationError",
    "RateLimitError",
    "MemoryNotFoundError",
    "ProcessingError",
    "TimeoutError",
    "ResourceExhaustedError",
    "ExceptionHandler",
    # Validators
    "DataValidator",
    "MemoryValidator",
    # Helpers
    "StringUtils",
    "DateTimeUtils",
    "JsonUtils",
    "FileUtils",
    "RetryUtils",
    "PerformanceUtils",
    "AsyncUtils",
    # Logging
    "LoggingManager",
    "get_logger",
]

# Add agents only if available
if _AGENTS_AVAILABLE:
    _all_components.extend(["MemoryAgent", "MemorySearchEngine"])

__all__ = _all_components
