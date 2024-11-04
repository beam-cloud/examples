from openai import OpenAI

openai_api_key = "YOUR_BEAM_TOKEN"
openai_api_base = "https://phi-vllm-c992edf-latest.app.beam.cloud"
openai_api_version = "v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=f"{openai_api_base}/{openai_api_version}",
)


def phi_chat() -> None:
    conversation_history = []

    print("Welcome to the CLI Chat Application!")
    print("Type 'quit' to exit the conversation. Image link is optional.")

    models = client.models.list()
    model = models.data[0].id
    print("Model is ready")

    try:
        while True:
            user_input = input("Question: ")
            img_link = input("Image link: ")

            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            if img_link:
                conversation_history.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_input},
                            {"type": "image_url", "image_url": {"url": img_link}},
                        ],
                    }
                )
            else:
                conversation_history.append({"role": "user", "content": user_input})

            response = client.chat.completions.create(
                model=model, messages=conversation_history
            )
            assistant_reply = response.choices[0].message.content
            conversation_history.append(
                {"role": "assistant", "content": assistant_reply}
            )

            print("Assistant:", assistant_reply)

    except KeyboardInterrupt:
        print("\nExiting the chat.")


if __name__ == "__main__":
    phi_chat()
