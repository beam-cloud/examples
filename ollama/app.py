from beam import endpoint, Image
from pydantic import BaseModel


class Answer(BaseModel):
    json: dict


image = (
    Image(python_version="python3.11")
    .add_python_packages(
        [
            "git+https://github.com/huggingface/transformers",
            "numpy<2",
            "fastapi[standard]==0.115.4",
            "pydantic==2.9.2",
            "starlette==0.41.2",
            "torch==2.4.0",
            "ollama",
        ]
    )
    .add_commands(
        [
            "curl -fsSL https://ollama.com/install.sh | sh",
        ]
    )
)


def load_model():
    import subprocess
    import time

    subprocess.Popen(["ollama", "serve"])
    time.sleep(5)
    subprocess.run(["ollama", "pull", "Osmosis/Osmosis-Structure-0.6B"], check=True)


@endpoint(
    name="ollamap-osmosis-structure", image=image, cpu=12, memory="32Gi", gpu="A10G", on_start=load_model
)
def generate(**inputs):
    from ollama import chat

    messages = inputs.get("messages", "")

    response = chat(
        messages=messages,
        model="Osmosis/Osmosis-Structure-0.6B",
        format=Answer.model_json_schema(),
    )

    answer = Answer.model_validate_json(response.message.content)
    print(answer)
    return {"answer": answer.json}
