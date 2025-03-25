from beam import Image, Pod

image = (
    Image(python_version="python3.11")
    .add_python_packages(
        [
            "transformers==4.47.1",
            "numpy<2",
            "fastapi[standard]==0.115.4",
            "pydantic==2.9.2",
            "starlette==0.41.2",
            "torch==2.4.0",
        ]
    )
    .add_commands(
        [
            'pip install "sglang[all]==0.4.1" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/'
        ]
    )
)

sglang_server = Pod(
    image=image,
    ports=[8080],
    cpu=12,
    memory="32Gi",
    gpu="A100-80",
    secrets=["HF_TOKEN"],
    entrypoint=[
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        "Qwen/Qwen2.5-7B-Instruct",
        "--port",
        "8080",
        "--host",
        "0.0.0.0",
    ],
)

res = sglang_server.create()

print("✨ SGlang server hosted at:", res.url)
