# Framework Integrations

Memoria works seamlessly with popular AI frameworks:

| Framework | Description | Example |
|-----------|-------------|---------|
| [Agno](https://github.com/Founder-ArcaFutura/Memoria/blob/main/examples/integrations/agno_example.py) | Memory-enhanced agent framework integration with persistent conversations | Simple chat agent with memory search |
| [CrewAI](https://github.com/Founder-ArcaFutura/Memoria/blob/main/examples/integrations/crewai_example.py) | Multi-agent system with shared memory across agent interactions | Collaborative agents with memory |
| [OpenAI Agent](https://github.com/Founder-ArcaFutura/Memoria/blob/main/examples/integrations/openai_agent_example.py) | Memory-enhanced OpenAI Agent with function calling and user preference tracking | Interactive assistant with memory search and user info storage |
| [Digital Ocean AI](https://github.com/Founder-ArcaFutura/Memoria/blob/main/examples/integrations/digital_ocean_example.py) | Memory-enhanced customer support using Digital Ocean's AI platform | Customer support assistant with conversation history |
| [LangChain](https://github.com/Founder-ArcaFutura/Memoria/blob/main/examples/integrations/langchain_example.py) | Enterprise-grade agent framework with advanced memory integration | AI assistant with LangChain tools and memory |
| [Swarms](https://github.com/Founder-ArcaFutura/Memoria/blob/main/examples/integrations/swarms_example.py) | Multi-agent system framework with persistent memory capabilities | Memory-enhanced Swarms agents with auto/conscious ingestion |

## Provider Helpers

For direct SDK usage, Memoria exposes optional client helpers. The
[`openai_client` module](./openai_client.md) demonstrates how to build a
`create_*_client` helper that wires a provider's official SDK into Memoria's
interceptor system. Downstream users can mirror this pattern to support other
providers while keeping the core package provider-agnostic.
