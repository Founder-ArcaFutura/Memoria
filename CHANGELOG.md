# Changelog

All notable changes to Memoria will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_No unreleased changes._

## [0.9.0a0] - 2025-10-14

> _Note: This release was previously tracked as 2.1.0 before the project reverted to the 0.9 alpha series._

### Added
- Documented and exposed per-task LLM routing controls via configuration and the
  `memoria assign-task-model` CLI helper so operators can steer ingestion and
  planning tasks without code changes.
- Documented the PolicyClient SDK helper so operators can automate governance workflows via the API.
- Automated structural migrations after CLI bootstrap/init-db and on server
  startup, surfacing the latest status in both the CLI and dashboard.
- Added provider capability checks that detect missing OpenAI, Gemini, or sync
  extras and present actionable installation or disablement guidance in the CLI
  and UI.

### Fixed
- Prevented duplicate Google Gemini configuration entries during provider setup so the initialization log only appears once.

### Changed
- Moved standalone example scripts into the `examples/` directory and archived scratch assets to keep the repository root focused on supported tooling.

## [0.9.0a0-pre1] - 2025-10-01

> _Note: This entry preserves the work shipped as 2.0.0 prior to the downgrade to the 0.9 alpha track._

### Added
- Introduced team and workspace multi-tenancy with new models, migrations, CLI routes, and API endpoints so memories, conversations, and namespaces can be scoped per organization or workspace.【F:memoria/config/settings.py†L331-L459】【F:memoria_server/api/memory_routes.py†L200-L520】
- Delivered pluggable synchronization backends, including Redis and PostgreSQL implementations with replication controls, enabling real-time propagation of memory changes across deployments.【F:memoria/sync/redis.py†L1-L137】【F:memoria/sync/postgres.py†L1-L200】【F:memoria/storage/service.py†L2012-L2056】
- Enabled vector search with embedding utilities, CLI helpers, and fuzzy matching backed by the new `rapidfuzz` dependency to improve retrieval quality when semantic matches are needed.【F:memoria/utils/embeddings.py†L1-L137】【F:memoria/database/search_service.py†L60-L146】【F:pyproject.toml†L38-L45】
- Implemented a memory relationship graph and candidate discovery pipeline that surfaces linked memories for promotion, analytics, and follow-up retrievals.【F:memoria/database/search_service.py†L2334-L2534】【F:memoria/storage/service.py†L1983-L2056】
- Expanded provider integrations with Anthropic Claude, Google Gemini, and configurable multi-provider routing so different tasks can target purpose-built model clients.【F:memoria/core/providers.py†L420-L604】【F:memoria/config/settings.py†L200-L318】
- Added analytics APIs and dashboard utilities for tracking category distributions, retention trends, usage frequency, and cluster health.【F:memoria/database/analytics.py†L1-L200】【F:memoria_server/api/utility_routes.py†L1-L200】
- Added a bulk import companion script and storage service hooks to validate and ingest exported memories with namespace remapping support.【F:scripts/import_memories.py†L1-L164】【F:memoria/storage/service.py†L1983-L2056】
- Added bootstrap migration utilities that ensure existing deployments populate `cluster_members` token and character columns automatically.【F:scripts/migrations/add_cluster_member_token_columns.py†L1-L53】【F:tests/test_missing_column_repair.py†L1-L65】

### Changed
- Enabled runtime `/settings` updates to propagate context-injection and synchronization toggles immediately to active Memoria instances without requiring restarts.【F:memoria_server/api/app_factory.py†L340-L394】【F:memoria_server/api/settings_routes.py†L360-L440】
- Replaced static context limits with adaptive orchestration that plans context window sizes based on analytics, privacy, and query complexity for each request.【F:memoria/core/context_orchestration.py†L1-L181】【F:memoria/core/context_injection.py†L1-L148】
- Updated open-source documentation to reflect shipped capabilities (vector search, provider adapters, relationship graph, analytics, sync backends) and clarified the public roadmap for upcoming work.【F:docs/open-source/features.md†L1-L120】

### Fixed
- Normalized legacy schema repairs so JSON columns receive dialect-appropriate storage (TEXT on SQLite, JSONB on PostgreSQL) while ensuring missing embedding columns are restored during initialization.【F:memoria/database/sqlalchemy_manager.py†L331-L414】【F:tests/test_missing_column_repair.py†L1-L83】
- Guarded SQLite backup and restore helpers against in-memory databases and normalized path handling when refreshing application bindings.【F:memoria/database/sqlalchemy_manager.py†L1738-L1779】【F:memoria_server/api/app_factory.py†L160-L236】
- Corrected StorageService context retrieval to honor team-scoped namespaces and avoid duplicate context rows when searching within collaborative spaces.【F:memoria/storage/service.py†L1523-L1594】【F:tests/test_retrieve_context_search.py†L1-L120】
- Allowed chat ingestion and storage validators to preserve HTML-like content safely while continuing to sanitize dangerous payloads.【F:memoria/utils/input_validator.py†L184-L212】【F:tests/test_chat_id_linking.py†L1-L120】
- Resolved time-range retrieval edge cases so fractional `x_coord` bounds filter memories consistently across short- and long-term tables.【F:memoria/storage/service.py†L2012-L2046】【F:tests/test_time_range_memory.py†L1-L80】
- Fixed OpenAI client creation to forward configuration kwargs correctly for standard, Azure, and custom deployments.【F:memoria/integrations/openai_client.py†L50-L105】【F:tests/test_memoria_provider_configuration.py†L1-L160】

### Removed
- Dropped the legacy `emotional_intensity` columns from memory tables and archived the removal migration for operators upgrading older databases.【F:memoria/database/models.py†L162-L199】【F:scripts/migrations/archive/remove_emotional_intensity.py†L1-L43】

## [1.2.0] - 2025-08-03

### 🚀 **Dual-Mode Memory System - Revolutionary Architecture**

**Major Release**: Complete overhaul of memory injection system with two distinct modes - Conscious short-term memory and Auto dynamic search.

#### ✨ **New Memory Modes**

**🧠 Conscious Mode (`conscious_ingest=True`)**
- **Short-Term Working Memory**: Mimics human conscious memory with essential info readily available
- **Startup Analysis**: Conscious agent analyzes long-term memory patterns at initialization
- **Memory Promotion**: Automatically promotes 5-10 essential conversations from long-term to short-term storage
- **One-Shot Injection**: Injects working memory context ONCE at conversation start, no repetition
- **Essential Context**: Names, current projects, preferences, skills always accessible

**🔍 Auto Mode (`auto_ingest=True`)**
- **Dynamic Database Search**: Uses retrieval agent for intelligent full-database search
- **Query Analysis**: AI-powered query understanding with OpenAI Structured Outputs
- **Continuous Retrieval**: Searches and injects 3-5 relevant memories on EVERY LLM call
- **Performance Optimized**: Caching, async processing, background threading
- **Full Coverage**: Searches both short-term and long-term memory databases

**⚡ Combined Mode (`conscious_ingest=True, auto_ingest=True`)**
- **Best of Both Worlds**: Working memory foundation + dynamic search capability
- **Layered Context**: Essential memories + query-specific memories
- **Maximum Intelligence**: Comprehensive memory utilization

#### 🔧 **API Changes**

**New Parameters**
```python
memoria = Memoria(
    conscious_ingest=True,  # Short-term working memory (one-shot)
    auto_ingest=True,       # Dynamic database search (continuous)
    openai_api_key="sk-..."
)
```

**Mode Behaviors**
- **Conscious**: Analysis at startup → Memory promotion → One-shot context injection
- **Auto**: Query analysis → Database search → Context injection per call
- **Combined**: Startup analysis + Per-call search

#### 🏗️ **Architecture Improvements**

**Enhanced Agents**
- **Conscious Agent**: Smarter long-term → short-term memory promotion
- **Retrieval Agent**: Performance optimized with caching and async support
- **Memory Agent**: Improved Pydantic-based processing

**Performance Enhancements**
- **Query Caching**: 5-minute TTL cache for search plans to reduce API calls
- **Async Processing**: `execute_search_async()` for concurrent operations
- **Background Threading**: Non-blocking search execution
- **Thread Safety**: Proper locking mechanisms for concurrent access

#### 📚 **Documentation & Examples**

**Updated Examples**
- **`memoria_example.py`**: Complete conscious-ingest demonstration with detailed comments
- **`auto_ingest_example.py`**: New example showcasing dynamic memory retrieval
- **Enhanced Comments**: Detailed explanations of each mode's behavior

**Updated Documentation**
- **README.md**: Comprehensive dual-mode system explanation
- **Mode Comparisons**: Clear distinctions between conscious vs auto modes
- **Configuration Examples**: All possible mode combinations

#### 🎯 **Use Cases**

**Conscious Mode Perfect For:**
- Personal assistants needing user context
- Project-specific conversations requiring background knowledge
- Situations where essential info should always be available
- One-time context establishment scenarios

**Auto Mode Perfect For:**
- Dynamic Q&A systems
- Research assistants requiring specific memory retrieval
- Multi-topic conversations needing relevant context injection
- Performance-critical applications with intelligent caching

**Combined Mode Perfect For:**
- Comprehensive personal AI assistants
- Maximum context utilization scenarios
- Professional applications requiring both background and specific context

#### 🛠️ **Developer Experience**

**Simplified Configuration**
```json
{
  "agents": {
    "conscious_ingest": true,
    "auto_ingest": false,
    "openai_api_key": "sk-..."
  }
}
```

**Enhanced Logging**
- Detailed mode-specific logging
- Performance metrics for caching and search
- Background processing status updates

#### ⚡ **Breaking Changes**

**Behavioral Changes**
- `conscious_ingest=True` now works differently (one-shot vs continuous)
- Memory injection timing changed based on selected mode
- Context injection strategies optimized per mode

**New Dependencies**
- Enhanced async processing requirements
- Additional threading support for background operations

## [1.1.0] - 2025-08-03

### 🧠 **Enhanced Conscious Ingestion System**

Major improvements to the intelligent memory processing and context injection system.

#### ✨ New Features

**Conscious Agent System**
- **Background Analysis**: Automatic analysis of long-term memory patterns every 6 hours
- **Essential Memory Promotion**: Promotes key personal facts to short-term memory for immediate access
- **Intelligent Context Selection**: AI-powered identification of most relevant memories for context injection
- **Personal Identity Extraction**: Automatically identifies and prioritizes user identity, preferences, and ongoing projects

**Enhanced Context Injection**
- **Essential Conversations**: Priority context from promoted memories for immediate relevance
- **Smart Memory Retrieval**: Up to 5 most relevant memories automatically injected into conversations
- **Category-Aware Context**: Different context strategies for facts, preferences, skills, and rules
- **Reduced Token Usage**: More efficient context injection with summarized essential information

**Improved Memory Processing**
- **Pydantic-Based Agents**: Structured memory processing with OpenAI Structured Outputs
- **Multi-Dimensional Scoring**: Frequency, recency, and importance scoring for memory selection
- **Entity Relationship Mapping**: Enhanced entity extraction and relationship tracking
- **Advanced Categorization**: Improved classification of facts, preferences, skills, context, and rules

#### 🔧 API Enhancements

**Conscious Ingestion Control**
```python
memoria = Memoria(
    database_connect="sqlite:///memory.db",
    conscious_ingest=True,  # Enable intelligent background analysis
    openai_api_key="sk-..."
)
```

**Memory Retrieval Methods**
- `get_essential_conversations()` - Access promoted essential memories
- `trigger_conscious_analysis()` - Manually trigger background analysis
- `retrieve_context()` - Enhanced context retrieval with essential memory priority

#### 📊 Background Processing

**Conscious Agent Features**
- **Automated Analysis**: Runs every 6 hours to analyze memory patterns
- **Selection Criteria**: Personal identity, preferences, skills, current projects, relationships
- **Memory Promotion**: Automatically promotes essential conversations to short-term memory
- **Analysis Reasoning**: Detailed reasoning for memory selection decisions

#### 🎯 Context Injection Improvements

**Essential Memory Integration**
- Essential conversations always included in context
- Smart memory limit management (3 essential + 2 specific)
- Category-based context prioritization
- Improved relevance scoring for memory selection

#### 🛠️ Developer Experience

**Enhanced Examples**
- Updated `memoria_example.py` with conscious ingestion showcase
- New `memory_retrieval_example.py` demonstrating function calling integration
- Advanced configuration examples with conscious agent settings

## [1.0.0] - 2025-08-03

### 🎉 **Production-Ready Memory Layer for AI Agents**

Complete professional-grade memory system with modular architecture, comprehensive error handling, and configuration management.

### ✨ Core Features
- **Universal LLM Integration**: Works with ANY LLM library (LiteLLM, OpenAI, Anthropic)
- **Pydantic-based Intelligence**: Structured memory processing with validation
- **Automatic Context Injection**: Relevant memories automatically added to conversations
- **Multiple Memory Types**: Short-term, long-term, rules, and entity relationships
- **Advanced Search**: Full-text search with semantic ranking

### 🏗️ Architecture
- **Modular Design**: Separated concerns with clear component boundaries
- **SQL Query Centralization**: Dedicated query modules for maintainability
- **Configuration Management**: Pydantic-based settings with auto-loading
- **Comprehensive Error Handling**: Context-aware exceptions with sanitized logging
- **Production Logging**: Structured logging with multiple output targets

### 🗄️ Database Support
- **Multi-Database**: SQLite, PostgreSQL, MySQL connectors
- **Query Optimization**: Indexed searches and connection pooling
- **Schema Management**: Version-controlled migrations and templates
- **Full-Text Search**: FTS5 support for advanced text search

### 🔧 Developer Experience
- **Type Safety**: Full Pydantic validation throughout
- **Simple API**: One-line enablement with `memoria.enable()`
- **Flexible Configuration**: File, environment, or programmatic setup
- **Rich Examples**: Basic usage, personal assistant, advanced config

### 📊 Memory Processing
- **Entity Extraction**: People, technologies, projects, skills
- **Smart Categorization**: Facts, preferences, skills, rules, context
- **Importance Scoring**: Multi-dimensional relevance assessment
- **Relationship Mapping**: Entity interconnections and memory graphs

### 🔌 Integrations
- **LiteLLM Native**: Uses official callback system (recommended)
- **OpenAI/Anthropic**: Clean wrapper classes for direct usage
- **Tool Support**: Memory search tools for function calling

### 🛡️ Security & Reliability
- **Input Sanitization**: Protection against injection attacks
- **Error Context**: Detailed error information without exposing secrets
- **Graceful Degradation**: Continues operation when components fail
- **Resource Management**: Automatic cleanup and connection pooling

### 📁 Project Structure
```
memoria/
├── core/              # Main memory interface and database
├── config/            # Configuration management system
├── agents/            # Pydantic-based memory processing
├── database/          # Multi-database support and queries
├── integrations/      # LLM provider integrations
├── utils/             # Helpers, validation, logging
└── tools/             # Memory search and retrieval tools
```

### 🎯 Philosophy Alignment
- **Second-memory for LLM work**: Never repeat context again
- **Flexible database connections**: Production-ready adapters
- **Simple, reliable architecture**: Just works out of the box
- **Conscious context injection**: Intelligent memory retrieval

### ⚡ Quick Start
```python
from memoria import Memoria

memoria = Memoria(
    database_connect="sqlite:///my_memory.db",
    conscious_ingest=True,
    openai_api_key="sk-..."
)
memoria.enable()  # Start recording all LLM conversations

# Use any LLM library - context automatically injected!
from litellm import completion
response = completion(model="gpt-4", messages=[...])
```

### 📚 Documentation
- Clean, focused README aligned with project vision
- Essential examples without complexity bloat
- Configuration guides for development and production
- Architecture documentation for contributors

### 🗂️ Archive Management
- Moved outdated files to `archive/` folder
- Updated `.gitignore` to exclude archive from version control
- Preserved development history while cleaning main structure

### 💡 Breaking Changes from Pre-1.0
- Moved from enum-driven to Pydantic-based processing
- Simplified API surface with focus on `enable()/disable()`
- Restructured package layout for better modularity
- Enhanced configuration system replaces simple parameters

---

*This release represents the culmination of the vision outlined in the original architecture documents, delivering a production-ready memory layer that "just works" for AI developers.*