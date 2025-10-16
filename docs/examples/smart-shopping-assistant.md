# Example: Smart Shopping Assistant with Azure AI

This example showcases an enterprise-grade, AI-powered shopping assistant that uses Memoria for persistent memory and integrates with **Azure AI Foundry**.

## Overview

This demo simulates a sophisticated shopping assistant that can:

-   **Learn Customer Preferences**: It remembers past purchases, brand affinities (e.g., Apple products), and budget constraints.
-   **Provide Personalized Recommendations**: Suggests products based on a customer's shopping history and stated needs.
-   **Maintain Context**: Keeps track of the conversation and shopping goals across multiple interactions.
-   **Handle Various Scenarios**: The demo runs through several predefined scenarios, including a first-time purchase, a return visit, gift shopping, and home improvement.

This example is designed to demonstrate how Memoria can be used in a corporate or enterprise environment, integrating with cloud services like Azure to power intelligent, memory-aware applications.

## How it Works

The demo, located in `demos/smart_shopping_assistant/smart_shopping_demo.py`, is a self-running script that simulates a series of customer interactions. It does not require user input.

The `SmartShoppingAssistant` class initializes Memoria with a `ProviderConfig` specifically for Azure. This tells Memoria to use Azure OpenAI for its language model needs.

### Key Memoria Features Used:

-   **`ProviderConfig.from_azure()`**: The demo shows how to configure Memoria to use a specific LLM provider, in this case, Azure OpenAI. This is crucial for enterprise environments where specific cloud services are mandated.
-   **`Memoria(conscious_ingest=True)`**: Enables working memory to keep track of the current shopping session.
-   **`create_memory_tool(memoria)`**: Creates a `search_memory` function that the assistant uses to look up the customer's past purchases and preferences.
-   **`memoria.record_conversation()`**: Each interaction with the customer is recorded, building a rich history that informs future recommendations.

The agent uses its memory to recall that a customer previously bought an iPhone and therefore recommends a MacBook when they later ask for a laptop.

## Running the Demo

1.  **Navigate to the demo directory:**
    ```bash
    cd demos/smart_shopping_assistant
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your Azure environment:**
    This demo requires access to Azure AI Foundry and Azure OpenAI. You must have an Azure account and have the necessary resources deployed. Configure your environment by creating a `.env` file with the following:
    ```env
    PROJECT_ENDPOINT="https://your-project.eastus2.ai.azure.com"
    AZURE_OPENAI_API_KEY="your_azure_openai_api_key"
    AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
    AZURE_OPENAI_DEPLOYMENT_NAME="your_deployment_name"
    AZURE_OPENAI_API_VERSION="2024-12-01-preview"
    ```
    You will also need to be authenticated with Azure, for example by running `az login`.

4.  **Run the demo script:**
    ```bash
    python smart_shopping_demo.py
    ```

The script will run through the predefined scenarios automatically, printing the conversation to the console and showing how the assistant uses memory to personalize its responses.