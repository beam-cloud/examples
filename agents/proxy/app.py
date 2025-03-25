"""
This script is used to deploy the Proxy Lite Endpoint on Beam.

You can deploy this app by running `python app.py`
"""

from beam import Image, Pod

server = Pod(
    image=Image(python_version="python3.12")
    .add_python_packages(
        [
            "vllm==0.7.2",
            "git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef",
        ]
    )
    .add_commands(['echo "127.0.0.1 localhost" >> /etc/hosts']),
    ports=[7860],
    cpu=12,
    gpu="A10G",
    gpu_count=2,
    memory="32Gi",
    entrypoint=[
        "sh",
        "-c",
        "python -m vllm.entrypoints.openai.api_server --model convergence-ai/proxy-lite-3b --trust-remote-code --tokenizer-pool-size 10 --max_model_len 16384 --limit-mm-per-prompt image=1 --enable-auto-tool-choice --tool-call-parser hermes --port 7860",
    ],
)

res = server.create()
print("✨ Proxy Lite Endpoint hosted at:", res.url)
