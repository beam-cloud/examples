import openai

client = openai.Client(
    base_url="https://35b937b9-1a70-4343-89d9-1125b1290e4d-8080.app.beam.cloud/v1",
    api_key="BEAM_API_KEY", # Make sure to replace it with your Beam API key not OpenAI API or Empty
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
