# Example: Personal Diary Assistant

This example demonstrates how to build a personal diary assistant using Memoria to store and analyze daily entries, track moods, and provide personalized insights.

## Overview

The Personal Diary Assistant is a Streamlit web application that acts as an intelligent diary. It allows users to:

-   **Record Daily Entries**: Log daily activities, thoughts, and feelings.
-   **Track Mood and Productivity**: Assign a mood and productivity score to each day.
-   **Analyze Patterns**: Use an AI agent to analyze trends in mood, habits, and productivity over time.
-   **Receive Personalized Recommendations**: Get actionable advice based on historical data.
-   **Search Past Entries**: Easily retrieve past memories using natural language.

This demo showcases how Memoria can provide long-term, persistent memory for a personal application, enabling it to learn about the user and offer increasingly personalized and relevant interactions.

## How it Works

The assistant is built using a combination of `streamlit` for the user interface and `litellm` to interact with an OpenAI model. The core logic resides in `demos/personal_diary_assistant/diary_assistant.py`.

The `PersonalDiaryAssistant` class initializes a `Memoria` instance, creating a dedicated SQLite database (`personal_diary.db`) and a namespace (`personal_diary`) to keep the diary entries isolated.

### Key Memoria Features Used:

-   **`Memoria(conscious_ingest=True)`**: Enables "conscious" or working memory, allowing the assistant to maintain context within a session.
-   **`memoria.enable()`**: Activates the memory system to automatically record interactions.
-   **`create_memory_tool(memoria)`**: Creates a tool that can be used by an AI agent (in this case, via `litellm`'s function calling) to search the memory.
-   **`memoria.record_conversation()`**: Stores each diary entry and interaction in the Memoria database.

The agent uses a `memory_search` function to query the user's history, allowing it to answer questions like "How has my mood been lately?" or "What were my goals last month?".

## Running the Demo

1.  **Navigate to the demo directory:**
    ```bash
    cd demos/personal_diary_assistant
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your environment:**
    Create a `.env` file in the directory with your OpenAI API key:
    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run streamlit_app.py
    ```

This will launch a web interface where you can interact with your personal diary assistant. You can also run the command-line version with `python diary_assistant.py`.