"""
Utils package for Memoria - Comprehensive utilities and helpers
"""

# Enhanced exception handling
from .cluster_index import (
    get_cluster_activity,
    get_heaviest_clusters,
    query_cluster_index,
)
from .exceptions import (
    AgentError,
    AuthenticationError,
    ConfigurationError,
    DatabaseError,
    ExceptionHandler,
    IntegrationError,
    MemoriaError,
    MemoryNotFoundError,
    ProcessingError,
    RateLimitError,
    ResourceExhaustedError,
    SecurityError,
    TimeoutError,
    ValidationError,
)

# Gravity-based ranking utilities
from .gravity import GravityScorer

# Helper utilities
from .helpers import (
    AsyncUtils,
    DateTimeUtils,
    FileUtils,
    JsonUtils,
    PerformanceUtils,
    RetryUtils,
    StringUtils,
)

# Logging utilities
from .logging import LoggingManager, get_logger

# Core Pydantic models
from .pydantic_models import (
    ConversationContext,
    EntityType,
    ExtractedEntities,
    MemoryCategory,
    MemoryCategoryType,
    MemoryImportance,
    ProcessedMemory,
    RetentionType,
)

# Validation utilities
from .validators import DataValidator, MemoryValidator

__all__ = [
    # Pydantic Models
    "ProcessedMemory",
    "MemoryCategory",
    "ExtractedEntities",
    "MemoryImportance",
    "ConversationContext",
    "MemoryCategoryType",
    "RetentionType",
    "EntityType",
    # Exceptions
    "MemoriaError",
    "DatabaseError",
    "AgentError",
    "ConfigurationError",
    "ValidationError",
    "IntegrationError",
    "AuthenticationError",
    "SecurityError",
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
    "GravityScorer",
    # Logging
    "LoggingManager",
    "get_logger",
    "query_cluster_index",
    "get_cluster_activity",
    "get_heaviest_clusters",
]
