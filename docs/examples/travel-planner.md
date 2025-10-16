# Example: Travel Planner Agent with CrewAI

This example demonstrates how to build a multi-agent travel planning system using **CrewAI**, with Memoria providing persistent, personalized memory for each user.

## Overview

This demo features a crew of AI agents that collaborate to plan personalized travel itineraries. The agents can:

-   **Research Destinations**: Use the SerperDevTool to find real-time information about flights, hotels, and activities.
-   **Remember Preferences**: Store and retrieve user travel preferences (e.g., budget, travel style, interests) using Memoria.
-   **Create Personalized Itineraries**: The planning agent uses the research and memories to craft a detailed, day-by-day travel plan.
-   **Provide Budget Advice**: A dedicated budget agent analyzes the plan and provides cost estimates and money-saving tips.

This example highlights how Memoria can serve as a shared memory layer for a team of autonomous agents, enabling them to work together more effectively and deliver a highly personalized user experience.

## How it Works

The demo, located in `demos/travel_planner/`, uses the `crewai` library to define a `Crew` of agents. The core logic in `travel_agent.py` sets up the agents and their tasks.

The `TravelPlannerAgent` class initializes Memoria and creates a `memory_search` tool. This tool is then provided to the CrewAI agents, allowing them to access the user's travel history and preferences during their planning process.

### Key Memoria Features Used:

-   **`Memoria(conscious_ingest=True)`**: Enables working memory for the travel planning session.
-   **`create_memory_tool(memoria)`**: Creates the `memory_search` tool that is passed to the CrewAI agents.
-   **`memoria.record_conversation()`**: Used to save the user's initial request and preferences, as well as the final generated travel plan, creating a memory of the trip for future interactions.

When a user asks for a trip to a destination they've inquired about before, the research agent can use the `memory_search` tool to retrieve their previously stated preferences, leading to a more personalized plan without the user having to repeat themselves.

## Running the Demo

1.  **Navigate to the demo directory:**
    ```bash
    cd demos/travel_planner
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your environment:**
    Create a `.env` file in this directory with your API keys for OpenAI and Serper:
    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    SERPER_API_KEY=your_serper_api_key_here
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

This will launch a web interface where you can describe your desired trip, set your preferences, and let the AI crew plan it for you.