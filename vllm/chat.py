import json
import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage

from transformers import AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress
from rich.live import Live


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
        self, user_input: str, img_link: Optional[str] = None, stream: bool = False
    ) -> str:
        """Process user input and return assistant's response."""
        if self.model == "OpenGVLab/InternVL2_5-8B" and img_link:
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
            stream=stream,
        )

        if stream:
            print("Assistant: ", end="", flush=True)
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
            print()
            self.conversation_history.append(
                {"role": "assistant", "content": full_response}
            )
            return full_response

        else:
            assistant_message = response.choices[0].message
            final_response = ""
            if self.supports_tools() and assistant_message.tool_calls:
                final_response = self.handle_tool_calls(assistant_message)
            else:
                final_response = assistant_message.content
            self.conversation_history.append(
                {"role": "assistant", "content": final_response}
            )
            return final_response


console = Console()


def chat() -> None:
    """Main chat loop"""
    console.print(
        Panel(
            "[bold cyan]Welcome to the CLI Chat Application![/bold cyan]", expand=False
        )
    )
    console.print("[bold white]Type 'quit' to exit the conversation.[/bold white]")

    # Setup client
    app_url = Prompt.ask("[bold yellow]Enter the app URL[/bold yellow]")
    beam_config = open(f"{os.path.expanduser('~')}/.beam/config.ini").read().split("\n")
    if len(beam_config) < 2:
        console.print(
            "[bold red]Error:[/bold red] Beam config does not contain a token.",
            style="red",
        )
        return
    beam_token = beam_config[1].split(" = ")[1].strip()

    stream = Prompt.ask("[bold yellow]Stream mode? (y/n)[/bold yellow]").lower() == "y"

    client = OpenAI(
        api_key=beam_token,
        base_url=f"{app_url}/v1",
    )

    models = client.models.list()
    model = models.data[0].id
    console.print(Panel(f"✅ [bold green]Model {model} is ready[/bold green]"))

    chat_app = ChatApplication(client, model)

    try:
        while True:
            console.print("\n" + "-" * 50, style="dim")
            user_input = Prompt.ask("[bold blue]Question[/bold blue]")
            if user_input.lower() == "quit":
                console.print("[bold magenta]Goodbye! 👋[/bold magenta]")
                break

            # Handle image input for vision models
            img_link = None
            if model == "OpenGVLab/InternVL2_5-8B":
                img_link = Prompt.ask(
                    "[bold yellow]Image link (press enter to skip)[/bold yellow]"
                )

            chat_app.conversation_history.append(
                {"role": "user", "content": user_input}
            )

            # Start timer
            start_time = time.time()

            if stream:
                full_response = ""
                try:
                    with Live(
                        Panel(
                            "⏳ [bold cyan]Thinking...[/bold cyan]",
                            title="Assistant",
                            expand=False,
                            style="cyan",
                        ),
                        console=console,
                        refresh_per_second=10,
                    ) as live:
                        response = chat_app.client.chat.completions.create(
                            model=chat_app.model,
                            messages=chat_app.conversation_history,
                            stream=True,
                        )

                        for chunk in response:
                            if hasattr(chunk.choices[0].delta, "content"):
                                content = chunk.choices[0].delta.content
                                if content:
                                    full_response += content
                                    live.update(
                                        Panel(
                                            full_response,
                                            title="Assistant",
                                            expand=False,
                                            style="cyan",
                                        )
                                    )

                    # Append final assistant message
                    chat_app.conversation_history.append(
                        {"role": "assistant", "content": full_response}
                    )

                except Exception as e:
                    console.print(f"[bold red]Error during streaming:[/bold red] {e}")

            else:
                # Non-streaming mode (blocking response)
                try:
                    with Progress() as progress:
                        task = progress.add_task(
                            "[cyan]Generating response...", total=100
                        )
                        response = chat_app.process_user_input(
                            user_input, img_link, stream
                        )
                        progress.update(task, advance=100)

                    chat_app.conversation_history.append(
                        {"role": "assistant", "content": response}
                    )

                    console.print("\n✨ [bold green]Final Response:[/bold green] ✨")
                    console.print(
                        Panel(response, title="Assistant", expand=False, style="cyan")
                    )

                except Exception as e:
                    console.print(
                        f"[bold red]Error during response generation:[/bold red] {e}"
                    )

            # Measure tokens
            tokenizer = AutoTokenizer.from_pretrained(model)
            output_tokens = len(tokenizer.encode(full_response if stream else response))

            end_time = time.time()
            total_time = end_time - start_time
            output_tokens_per_second = (
                output_tokens / total_time if total_time > 0 else 0
            )

            console.print(
                f"📜 [bold yellow]Tokens Generated:[/bold yellow] {output_tokens}"
            )
            console.print(
                f"⏳ [bold yellow]Time Taken:[/bold yellow] {total_time:.2f}s"
            )
            console.print(
                f"⚡ [bold yellow]Tokens Per Second:[/bold yellow] {output_tokens_per_second:.2f}"
            )

    except KeyboardInterrupt:
        console.print("\n[bold red]Exiting the chat.[/bold red]")


if __name__ == "__main__":
    chat()
