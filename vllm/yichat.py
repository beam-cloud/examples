from openai import OpenAI


def chat_with_gpt():
    openai_completions_api_version = "v1"
    beam_deployment_url = "https://yicoder-vllm-ea7443c-latest.app.beam.cloud"

    client = OpenAI(
        api_key="YOUR_BEAM_TOKEN",
        base_url=f"{beam_deployment_url}/{openai_completions_api_version}",
    )

    conversation_history = []

    print("Welcome to the CLI Chat Application!")
    print("Type 'quit' to exit the conversation.")

    if client.models.list().data[0].id == "01-ai/Yi-Coder-9B-Chat":
        print("Model is ready")
    else:
        print("Failed to load model")
        exit(1)

    try:
        while True:
            user_input = input("You: ")

            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            conversation_history.append({"role": "user", "content": user_input})
            response = client.chat.completions.create(
                model="01-ai/Yi-Coder-9B-Chat", messages=conversation_history
            )
            assistant_reply = response.choices[0].message.content
            conversation_history.append(
                {"role": "assistant", "content": assistant_reply}
            )

            print("Assistant:", assistant_reply)

    except KeyboardInterrupt:
        print("\nExiting the chat.")


if __name__ == "__main__":
    chat_with_gpt()
