from beam import Pod, Image

image = (
    Image(python_version="python3.11")
    .add_commands(
        [
            "apt update && apt install -y git libgl1-mesa-glx libglib2.0-0",
        ]
    )
    .add_python_packages(
        [
            "torch==2.4.1",
        ]
    )
    .add_python_packages(
        [
            "torchvision",
            "xformers",
            "tensorboard",
            "gfpgan",
            "lpips",
            "realesrgan",
            "gdown",
            "mediapipe",
            "pytorch_lightning",
            "git+https://github.com/huggingface/transformers",
        ]
    )
    .add_commands(
        [
            "git clone https://github.com/philz1337x/clarity-upscaler.git /clarity-upscaler",
            "cd /clarity-upscaler && python download_weights.py",
            "pip install --upgrade setuptools",
        ]
    )
).build_with_gpu("A10G")

clarity_upscaler_server = Pod(
    image=image,
    ports=[7861],
    cpu=14,
    memory="32Gi",
    gpu="A10G",
    entrypoint=[
        "python",
        "/clarity-upscaler/webui.py",
        "--listen",
        "--vae-path",
        "/clarity-upscaler/models/VAE/vae-ft-mse-840000-ema-pruned.safetensors",
    ],
)

res = clarity_upscaler_server.create()
print("Clarity Upscaler server created:", res.url)
