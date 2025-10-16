# Example: Researcher Agent with Agno

This example demonstrates how to build a powerful AI research assistant using the **Agno** agent framework, with Memoria providing persistent memory.

## Overview

The Researcher Agent can perform real-time web research, generate comprehensive reports, and, most importantly, remember its findings across sessions. This allows the agent to build upon previous research, avoid redundant work, and provide more contextually aware answers over time.

The demo includes two types of agents:

-   **Research Agent**: Conducts web searches using the Exa API, analyzes the results, and generates a formatted report.
-   **Memory Assistant Agent**: Interacts with the user to retrieve, summarize, and organize findings from previous research sessions stored in Memoria.

This example showcases how Memoria can be used to give agentic systems a persistent, queryable memory, making them more efficient and intelligent.

## How it Works

The core logic, found in `demos/researcher_agent/researcher.py`, defines a `Researcher` class that manages the Memoria instance and the Agno agents.

A key feature of this demo is the `run_agent_with_memory` method. It wraps the agent's execution cycle to automatically record each step of the conversation—including internal thoughts and tool calls—into Memoria. This provides a rich, auditable history of the agent's reasoning process.

### Key Memoria Features Used:

-   **`Memoria(conscious_ingest=True, auto_ingest=True)`**: Enables both working memory for session context and dynamic memory for retrieving relevant information during a task.
-   **`create_memory_tool(memoria)`**: Creates a `memory_search` tool that the Agno agents can use to query their own history.
-   **`memoria.record_conversation()`**: Called within a custom wrapper to automatically log the agent's interactions, creating a persistent memory of its research.

## Running the Demo

1.  **Navigate to the demo directory:**
    ```bash
    cd demos/researcher_agent
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your environment:**
    Create a `.env` file in this directory with your API keys for OpenAI and Exa:
    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    EXA_API_KEY=your_exa_api_key_here
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

This will launch a web interface where you can chat with the research agent or the memory assistant.