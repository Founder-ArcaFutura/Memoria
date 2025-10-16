import json

import litellm
from dotenv import load_dotenv

from memoria import ConfigManager, Memoria, create_memory_tool

load_dotenv()  # Load environment variables from .env file

# Load model configuration
# Users can change the model by setting `agents.default_model` in memoria.json:
# {
#   "agents": {"default_model": "gpt-4o-mini"}
# }
config = ConfigManager()
config.auto_load()
MODEL_NAME = config.get_setting("agents.default_model", "gpt-4o")

# Create your workspace memory (without automatic injection)
office_work = Memoria(
    database_connect="sqlite:///office_memory.db",
    # conscious_ingest=False,  # Disable background conscious analysis
    verbose=True,  # Enable verbose logging
)

office_work.enable()  # Start recording conversations

# Create memory tool for LLM function calling
memory_tool = create_memory_tool(office_work)
tool_schema = {"type": "function", "function": memory_tool.get_tool_schema()}

# LLMs can call the tool for spatial retrieval like:
# {
#     "name": "memoria_memory",
#     "arguments": {
#         "operation": "spatial",
#         "x": 0.0,
#         "y": 0.0,
#         "z": 0.0,
#         "max_distance": 3.0
#     }
# }

# Use LiteLLM with function calling

# System prompt
SYSTEM_PROMPT = (
    "You are an AI assistant with memory capabilities. "
    "Use the memoria_memory tool to search or retrieve memories when needed."
)


def memory_query(**kwargs):
    """Execute memory tool with provided parameters"""
    try:
        return memory_tool.execute(**kwargs)
    except Exception as e:
        return f"Error: {str(e)}"


# Tools definition
tools = [tool_schema]


def chat_with_memory():
    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("🧠 AI Assistant with Memory Tools")
    print("Ask me anything! I can remember our conversations and learn about you.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! Our conversation has been saved to memory.")
            break

        # Add user message to conversation
        conversation_history.append({"role": "user", "content": user_input})

        try:
            # Make LLM call with function calling
            response = litellm.completion(
                model=MODEL_NAME,
                messages=conversation_history,
                tools=tools,
                verbose=True,  # Enable verbose logging
                tool_choice="auto",  # auto is default, but we'll be explicit
            )

            response_message = response.choices[0].message
            tool_calls = response.choices[0].message.tool_calls

            # Handle function calls
            if tool_calls:
                conversation_history.append(response_message)

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    if function_name == memory_tool.tool_name:
                        function_response = memory_query(**function_args)
                        op = function_args.get("operation", "search")
                        if op == "spatial":
                            dist = function_args.get("max_distance", 5.0)
                            print(
                                f"📍 Memory Tool: spatial -> Searching within {dist} units"
                            )
                        else:
                            query = function_args.get("query", "")
                            print(
                                f"🔍 Memory Tool: search -> Found results for '{query}'"
                            )

                        conversation_history.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": function_response,
                            }
                        )

                # Get final response after function calls
                final_response = litellm.completion(
                    model=MODEL_NAME, messages=conversation_history
                )

                final_content = final_response.choices[0].message.content
                print(f"AI: {final_content}")
                conversation_history.append(
                    {"role": "assistant", "content": final_content}
                )

            else:
                # No function calls, just respond normally
                content = response_message.content
                print(f"AI: {content}")
                conversation_history.append({"role": "assistant", "content": content})

        except Exception as e:
            print(f"Error: {e}")
            break


if __name__ == "__main__":
    chat_with_memory()
