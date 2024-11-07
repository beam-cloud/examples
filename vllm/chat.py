import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage


def get_current_weather(city: str, state: str, unit: str) -> str:
    """Mock weather function that returns a fixed response."""
    return f"The weather in {city}, {state} is 75 degrees {unit}. It is sunny with light clouds."


# Tool definitions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for, e.g. 'San Francisco'",
                    },
                    "state": {
                        "type": "string",
                        "description": "the two-letter abbreviation for the state that the city is in",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "state", "unit"],
            },
        },
    }
]

available_tools = {"get_current_weather": get_current_weather}


class ChatApplication:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.conversation_history: List[Dict[str, Any]] = []

    def handle_tool_calls(self, assistant_message: ChatCompletionMessage) -> str:
        """Handle tool calls and return the processed response."""
        responses = []

        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            if function_name not in available_tools:
                continue

            try:
                function_args = json.loads(tool_call.function.arguments)
                function_response = available_tools[function_name](**function_args)
                responses.append(function_response)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Error processing tool call: {e}")
                continue

        # Return the first response (since we're only handling weather requests)
        return responses[0] if responses else "I couldn't process that request."

    def supports_tools(self) -> bool:
        """Check if the current model supports tool calling."""
        return self.model == "mistralai/Mistral-7B-Instruct-v0.3"

    def process_user_input(
        self, user_input: str, img_link: Optional[str] = None
    ) -> str:
        """Process user input and return assistant's response."""
        if self.model == "microsoft/Phi-3.5-vision-instruct" and img_link:
            self.conversation_history.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_input},
                        {"type": "image_url", "image_url": {"url": img_link}},
                    ],
                }
            )
        else:
            self.conversation_history.append({"role": "user", "content": user_input})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            tools=tools if self.supports_tools() else None,
        )

        assistant_message = response.choices[0].message
        final_response = ""

        # Handle tool calls if present and supported
        if self.supports_tools() and assistant_message.tool_calls:
            final_response = self.handle_tool_calls(assistant_message)
        else:
            final_response = assistant_message.content

        self.conversation_history.append(
            {"role": "assistant", "content": final_response}
        )

        return final_response


def chat() -> None:
    """Main chat loop."""
    print("Welcome to the CLI Chat Application!")
    print("Type 'quit' to exit the conversation.")

    # Setup client
    app_url = input("Enter the app URL: ")
    beam_config = open(f"{os.path.expanduser('~')}/.beam/config.ini").read().split("\n")
    if len(beam_config) < 2:
        raise EnvironmentError("Beam config does not contain a token.")
    beam_token = beam_config[1].split(" = ")[1].strip()

    client = OpenAI(
        api_key=beam_token,
        base_url=f"{app_url}/v1",
    )

    models = client.models.list()
    model = models.data[0].id
    print(f"Model {model} is ready")

    chat_app = ChatApplication(client, model)

    try:
        while True:
            user_input = input("Question: ")
            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            # Handle image input for vision models
            img_link = None
            if model == "microsoft/Phi-3.5-vision-instruct":
                img_link = input("Image link (press enter to skip): ")

            response = chat_app.process_user_input(user_input, img_link)
            print("Assistant:", response)

    except KeyboardInterrupt:
        print("\nExiting the chat.")


if __name__ == "__main__":
    chat()
