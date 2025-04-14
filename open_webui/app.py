from beam import Pod, Image

image = Image(
    python_version="python3.11",
).add_commands(
    [
        "pip install --ignore-installed blinker",
        "pip install open-webui",
    ]
)

BEAM_LLM_API_BASE_URL = ""  # Replace with your Beam LLM API base URL
BEAM_API_KEY = ""  # Replace with your Beam API key

webui_server = Pod(
    name="open-webui",
    cpu=12,
    memory="32Gi",
    gpu="A10G",
    ports=[8080],
    image=image,
    env={
        "OPENAI_API_BASE_URL": BEAM_LLM_API_BASE_URL,
        "OPENAI_API_KEY": BEAM_API_KEY,
    },
    entrypoint=["sh", "-c", "open-webui serve"],
)

result = webui_server.create()
print("URL:", result.url)
