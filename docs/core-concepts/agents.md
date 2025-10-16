# Agent System Documentation

Memoria features a sophisticated multi-agent system for intelligent memory processing with dual memory modes and provider configuration support.

## Overview

The agent system consists of three specialized AI agents that work together to provide intelligent memory management:

1. **Memory Agent** - Processes every conversation with structured outputs using Pydantic models
2. **Conscious Agent** - Copies conscious-info labeled memories to short-term memory for immediate access
3. **Memory Search Engine** - Intelligently retrieves and injects relevant context based on user queries

## Dual Memory Modes

Memoria introduces two distinct memory modes that can be used separately or together:

### Conscious Ingest Mode (`conscious_ingest=True`)
- **One-shot Context Injection**: Injects context at conversation start
- **Essential Memory Promotion**: Copies conscious-info labeled memories to short-term memory
- **Background Processing**: Runs once at program startup
- **Use Case**: Persistent context for entire conversation sessions

### Auto Ingest Mode (`auto_ingest=True`)
- **Real-time Context Injection**: Injects relevant memories on every LLM call
- **Dynamic Retrieval**: Uses Memory Search Engine to find contextually relevant memories
- **Intelligent Search**: Analyzes user input to retrieve the most appropriate memories
- **Use Case**: Dynamic, query-specific memory injection

## Memory Agent

The Memory Agent is responsible for processing every conversation and extracting structured information using OpenAI's Structured Outputs with Pydantic models. It supports multiple provider configurations for flexibility.

### Provider Configuration

The Memory Agent can be configured with various LLM providers:

```python
from memoria import Memoria
from memoria.core.providers import ProviderConfig

# Azure OpenAI configuration
azure_config = ProviderConfig.from_azure(
    api_key="your-azure-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    azure_deployment="gpt-4o",
    api_version="2024-02-01"
)

memoria = Memoria(
    database_connect="sqlite:///memory.db",
    provider_config=azure_config,
    conscious_ingest=True
)
```

### Functionality

- **Categorization**: Classifies information as fact, preference, skill, context, or rule
- **Entity Extraction**: Identifies people, technologies, topics, skills, projects, and keywords
- **Importance Scoring**: Determines retention type (short-term, long-term, permanent)
- **Content Generation**: Creates searchable summaries and optimized text
- **Storage Decisions**: Determines what information should be stored and why

### Categories

| Category | Description | Examples |
|----------|-------------|----------|
| **fact** | Factual information, definitions, technical details | "I use PostgreSQL for databases" |
| **preference** | User preferences, likes/dislikes, personal choices | "I prefer clean, readable code" |
| **skill** | Skills, abilities, competencies, learning progress | "Experienced with FastAPI" |
| **context** | Project context, work environment, current situations | "Working on e-commerce project" |
| **rule** | Rules, policies, procedures, guidelines | "Always write tests first" |

### Retention Guidelines

- **short_term**: Recent activities, temporary information (expires ~7 days)
- **long_term**: Important information, learned skills, preferences
- **permanent**: Critical rules, core preferences, essential facts

## Conscious Agent

The Conscious Agent is responsible for copying conscious-info labeled memories from long-term memory directly to short-term memory for immediate context availability.

### Background Processing

**Execution**: Runs once at program startup when `conscious_ingest=True`

**Function**: Copies all memories with `conscious-info` labels to short-term memory

**Purpose**: Provides persistent context throughout the conversation session

### How It Works

1. **Startup Analysis**: Scans all long-term memories for conscious-info labels
2. **Memory Transfer**: Copies labeled memories to short-term memory
3. **Persistent Context**: These memories remain available for the entire session
4. **One-Shot Operation**: Runs once at initialization, not continuously

### Usage

```python
from memoria import Memoria

memoria = Memoria(
    database_connect="sqlite:///memory.db",
    conscious_ingest=True,  # Enable conscious agent
    verbose=True  # See conscious agent activity
)

memoria.enable()  # Triggers conscious agent startup
```

## Memory Search Engine

The Memory Search Engine (formerly Retrieval Agent) is responsible for intelligent memory retrieval and context injection, particularly for auto-ingest mode.

### Query Understanding

- **Intent Analysis**: Understands what the user is actually looking for
- **Parameter Extraction**: Identifies key entities, topics, and concepts  
- **Strategy Planning**: Recommends the best approach to find relevant memories
- **Filter Recommendations**: Suggests appropriate filters for category, importance, etc.

### Auto-Ingest Context Retrieval

When `auto_ingest=True`, the Memory Search Engine:

1. **Analyzes User Input**: Understands the context and intent of each query
2. **Searches Database**: Performs intelligent search across all memories
3. **Selects Relevant Context**: Returns 5 most relevant memories
4. **Injects Context**: Automatically adds context to the current conversation

### Usage with Auto-Ingest

```python
from memoria import Memoria

memoria = Memoria(
    database_connect="sqlite:///memory.db",
    auto_ingest=True,  # Enable auto-ingest mode
    verbose=True  # See search engine activity
)

# Every completion call will automatically include relevant context
from litellm import completion

response = completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What are my Python preferences?"}]
)
# Automatically includes relevant memories about Python preferences
```

### Search Strategies

The Memory Search Engine supports multiple search approaches:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **direct_database_search** | Primary method using database full-text search | Most reliable for keyword matching |
| **semantic_search** | AI-powered contextual understanding | Complex queries requiring inference |
| **keyword_search** | Direct keyword/phrase matching | Specific terms or technologies |
| **entity_search** | Search by entities (people, tech, topics) | "What did Mike say about React?" |
| **category_filter** | Filter by memory categories | "My preferences for code style" |
| **importance_filter** | Filter by importance levels | "Important information about project X" |
| **temporal_filter** | Search within specific time ranges | "Recent work on microservices" |

#### Fuzzy Search

When keyword and full-text searches return few results, Memoria can fall back to
fuzzy matching powered by [RapidFuzz](https://github.com/maxbachmann/RapidFuzz).
Enable fuzzy search by passing `use_fuzzy=True` and optional configuration
values:

```python
results = db_manager.search_memories(
    query="reccomend",  # misspelled
    namespace="default",
    use_fuzzy=True,
    fuzzy_min_similarity=50,
    fuzzy_max_results=5,
    adaptive_min_similarity=True,
)
```

* `fuzzy_min_similarity` (0-100): Base minimum similarity score required for a
  match.
* `fuzzy_max_results`: Maximum number of fuzzy matches to return.
* `adaptive_min_similarity`: Reduce the minimum similarity for longer queries
  (subtracts 2 points per extra token).

### Query Examples

- "What did I learn about X?" → Focus on facts and skills related to X
- "My preferences for Y" → Focus on preference category
- "Rules about Z" → Focus on rule category  
- "Recent work on A" → Temporal filter + context/skill categories
- "Important information about B" → Importance filter + keyword search

## Configuration

### Dual Memory Mode Setup

You can use either mode individually or combine them:

```python
from memoria import Memoria

# Conscious ingest only (one-shot context at startup)
memoria_conscious = Memoria(
    database_connect="sqlite:///memory.db",
    conscious_ingest=True,  # Enable conscious agent
    openai_api_key="sk-..."  # Required for memory processing
)

# Auto ingest only (dynamic context on every call)
memoria_auto = Memoria(
    database_connect="sqlite:///memory.db",
    auto_ingest=True,  # Enable auto-ingest mode
    openai_api_key="sk-..."  # Required for memory processing
)

# Both modes together (maximum intelligence)
memoria_combined = Memoria(
    database_connect="sqlite:///memory.db",
    conscious_ingest=True,  # Essential context at startup
    auto_ingest=True,       # Dynamic context per query
    openai_api_key="sk-..."  # Required for both agents
)

memoria_combined.enable()  # Start all enabled agents
```

### Provider Configuration

Configure different LLM providers for the agents:

```python
from memoria.core.providers import ProviderConfig

# OpenAI (default)
openai_config = ProviderConfig.from_openai(
    api_key="sk-...",
    model="gpt-4o"
)

# Azure OpenAI
azure_config = ProviderConfig.from_azure(
    api_key="your-azure-key",
    azure_endpoint="https://your-resource.openai.azure.com/",
    azure_deployment="gpt-4o",
    api_version="2024-02-01"
)

# Custom endpoint (Ollama, etc.)
custom_config = ProviderConfig.from_custom(
    base_url="http://localhost:11434/v1",
    api_key="not-required",
    model="llama3"
)

memoria = Memoria(
    database_connect="sqlite:///memory.db",
    provider_config=azure_config,  # Use any configuration
    conscious_ingest=True,
    auto_ingest=True
)
```

## Context Injection Strategy

### Conscious Ingest Mode

When `conscious_ingest=True`:

1. **Startup Analysis**: Conscious Agent scans for conscious-info labeled memories
2. **One-shot Transfer**: Transfers all labeled memories to short-term memory
3. **Session Persistence**: Context remains available throughout the session
4. **No Re-analysis**: Context stays fixed until next program restart

### Auto Ingest Mode

When `auto_ingest=True`:

1. **Per-Query Analysis**: Memory Search Engine analyzes each user input
2. **Dynamic Retrieval**: Searches entire database for relevant memories
3. **Context Selection**: Returns up to 5 most relevant memories
4. **Real-time Injection**: Injects context into each LLM call

### Combined Mode Strategy

When both modes are enabled:

```python
# Combined context injection
essential_context = conscious_agent.get_short_term_memories()  # Fixed context
dynamic_context = search_engine.retrieve_context(user_input)  # Query-specific context

# Both contexts are intelligently merged and injected
total_context = merge_contexts(essential_context, dynamic_context)
inject_context(total_context)
```

## Monitoring and Debugging

### Verbose Mode

Enable verbose logging to see agent activity:

```python
memoria = Memoria(
    database_connect="sqlite:///memory.db",
    conscious_ingest=True,
    auto_ingest=True,
    verbose=True  # Show all agent activity
)
```

### Log Messages

With `verbose=True`, you'll see:

**Memory Agent Activity**:
```
[MEMORY] Processing conversation: "I prefer FastAPI"
[MEMORY] Categorized as 'preference', importance: 0.8
[MEMORY] Extracted entities: {'technologies': ['FastAPI']}
```

**Conscious Agent Activity**:
```
[CONSCIOUS] Starting conscious ingest at startup
[CONSCIOUS] Found 5 conscious-info labeled memories
[CONSCIOUS] Copied 5 memories to short-term memory
[CONSCIOUS] Conscious ingest complete
```

**Memory Search Engine Activity**:
```
[AUTO-INGEST] Starting context retrieval for query: 'What are my Python preferences?'
[AUTO-INGEST] Direct database search returned 3 results
[AUTO-INGEST] Context injection successful: 3 memories
```

### Manual Control

```python
# Check agent status
print(f"Conscious ingest enabled: {memoria.conscious_ingest}")
print(f"Auto ingest enabled: {memoria.auto_ingest}")

# Get memory statistics
stats = memoria.get_memory_stats()
print(f"Total memories: {stats.get('total_memories', 0)}")

# Check short-term memory (conscious ingest)
if memoria.conscious_ingest:
    short_term = memoria.db_manager.get_short_term_memories(namespace=memoria.namespace)
    print(f"Short-term memories: {len(short_term)}")

# Test auto-ingest context retrieval
if memoria.auto_ingest:
    context = memoria._get_auto_ingest_context("What are my preferences?")
    print(f"Auto-ingest context: {len(context)} memories")
```

## Performance Considerations

### Token Usage

The dual agent system is designed to be token-efficient:

- **Structured Outputs**: Pydantic models reduce parsing overhead
- **Smart Context Limits**: Automatic limits prevent token overflow
- **Mode Selection**: Choose the right mode for your use case
- **Provider Flexibility**: Use cost-effective models like GPT-4o-mini

### Mode Comparison

| Feature | Conscious Ingest | Auto Ingest | Combined |
|---------|------------------|-------------|----------|
| **Context Type** | Fixed essential | Dynamic relevant | Both |
| **When Active** | Startup only | Every LLM call | Both |
| **Token Usage** | Low (one-time) | Medium (per call) | Higher |
| **Responsiveness** | Fast | Real-time | Fast + Real-time |
| **Best For** | Persistent context | Query-specific context | Maximum intelligence |

### Background Processing

- **Conscious Mode**: Minimal overhead (startup only)
- **Auto Mode**: Real-time processing with recursion protection
- **Provider Support**: All modes work with any configured provider
- **Graceful Degradation**: Continues working if agents fail

## Troubleshooting

### Common Issues

**No API Key**:
```
Memory Agent initialization failed: No API key provided
```
**Solution**: Configure provider or set OPENAI_API_KEY environment variable

**Auto-Ingest Recursion**:
```
Auto-ingest: Recursion detected, using direct database search
```
**Solution**: This is normal - the system prevents infinite loops automatically

**No Context Retrieved**:
```
Auto-ingest: Direct database search returned 0 results
```
**Solution**: Build up more memory data through conversations

**Conscious Ingest No Memories**:
```
ConsciousAgent: No conscious-info memories found
```
**Solution**: Label important memories with conscious-info or have more conversations

### Debug Commands

```python
# Check agent configuration
print(f"Conscious ingest: {memoria.conscious_ingest}")
print(f"Auto ingest: {memoria.auto_ingest}")
print(f"Provider: {memoria.provider_config.api_type if memoria.provider_config else 'Default'}")

# Test memory processing
try:
    # Test memory agent (if available)
    if hasattr(memoria, 'memory_agent'):
        print("Memory agent available")
    
    # Test context retrieval
    if memoria.auto_ingest:
        context = memoria._get_auto_ingest_context("test query")
        print(f"Auto-ingest working: {len(context)} results")
        
    # Check short-term memory
    if memoria.conscious_ingest:
        short_term = memoria.db_manager.get_short_term_memories(namespace=memoria.namespace)
        print(f"Short-term memories: {len(short_term)}")
        
except Exception as e:
    print(f"Agent test failed: {e}")

# Check memory statistics
stats = memoria.get_memory_stats()
for key, value in stats.items():
    print(f"{key}: {value}")
```

## Best Practices

### For Users

1. **Choose the Right Mode**: 
   - Use `conscious_ingest` for persistent context needs
   - Use `auto_ingest` for dynamic, query-specific context
   - Use both for maximum intelligence
   
2. **Label Important Memories**: Use conscious-info labels for essential context in conscious mode

3. **Be Specific**: Share clear information about yourself, preferences, and projects

4. **Be Consistent**: Use consistent terminology for technologies and concepts

5. **Share Context**: Mention your role, current projects, and goals

6. **Reference Previous**: Build on previous conversations naturally

### For Developers

1. **Provider Configuration**: Use ProviderConfig for flexible LLM provider setup

2. **API Key Management**: Always use environment variables for API keys

3. **Error Handling**: Implement graceful degradation when agents fail

4. **Monitoring**: Use verbose mode to understand agent behavior

5. **Testing**: Test with different conversation patterns and memory modes

6. **Resource Management**: Consider token usage when choosing between modes

## Agent Roadmap

### Recently Delivered
- **Relationship-aware retrieval**: The relationship graph now feeds agent prompts so conversations reference linked people, projects, and decisions automatically.
- **Vector-powered ranking**: Auto-ingest agents lean on the semantic search pipeline for higher-quality recall alongside rules-based filters.
- **Provider flexibility**: Adapters for OpenAI, Azure OpenAI, Anthropic, LiteLLM, and custom endpoints keep the core agents provider-agnostic.
- **Distributed readiness**: Sync backends propagate new memories and thread updates so concurrent agent workers stay in lockstep.
- **Team-aware deployments & migrations**: Team workspaces, namespace-aware migrations, and export/import flows are documented in the [team memory guide](../open-source/features.md#team-aware-deployments) and supported by the `memoria migrate` CLI plus the utilities under `scripts/migrations`.
- **Adaptive context orchestration**: The `ContextOrchestrator` module (`memoria/core/context_orchestration.py`) dynamically sizes prompts with analytics-driven limits and privacy floors, replacing the need for bespoke context-window tuning.
- **Multi-model routing**: Provider adapters for OpenAI, Azure, Anthropic, Gemini, LiteLLM, and custom endpoints—covered in the [LLM integration guide](../open-source/llms/overview.md)—ship with first-class orchestration for structured-output chains.

### Next Steps
- 🔭 **Provider-tuned context heuristics**: Wire the `provider_name` hooks exposed in `Memoria.prepare_context_window()` and `ContextOrchestrator.plan_initial_window()` to adjust limits per model family and unlock smarter budgeting for Claude, Gemini, and others.
- 🔭 **External search adapters**: Implement the `SearchStrategy.EXTERNAL` path in the database connector layer so retrieval agents can call out to managed search services when native FTS or embeddings miss.
- 🔭 **Additional ingest interceptors**: Extend the interceptor facade beyond the current LiteLLM callback so teams can plug in custom streaming/telemetry middlewares without forking the ingestion stack.
- 🔭 **Scenario-driven eval harness**: Expand the agent integration tests (see `tests/test_auto_ingest_system_prompt.py` and `tests/test_agent_permissions.py`) into a reusable evaluation runner that scores recall, precision, and privacy handling for contributed strategies.
