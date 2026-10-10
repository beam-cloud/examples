"""
This is a ComfyUI server that allows you to generate images using the Flux1 Schnell model.

To deploy this app, run "python app.py" in your terminal.
"""

from beam import Image, Pod

ORG_NAME = "Comfy-Org"
REPO_NAME = "flux1-schnell"
WEIGHTS_FILE = "flux1-schnell-fp8.safetensors"
COMMIT = "f2808ab17fe9ff81dcf89ed0301cf644c281be0a"

image = (
    Image()
    .add_commands(["apt update && apt install git -y"])
    .add_python_packages(
        [
            "fastapi[standard]==0.115.4",
            "comfy-cli==1.22.0",
            "huggingface_hub==2.2.0",
        ]
    )
    .add_commands(
        [
            "comfy --skip-prompt install --nvidia --version 0.39.0",
            f"hf download {ORG_NAME}/{REPO_NAME} {WEIGHTS_FILE} --revision {COMMIT} --local-dir /root/comfy/ComfyUI/models/checkpoints",
        ]
    )
)

comfyui_server = Pod(
    image=image,
    ports=[8000],
    cpu=12,
    memory="32Gi",
    gpu="RTX4090",
    entrypoint=["sh", "-c", "comfy launch -- --listen 0.0.0.0 --port 8000"],
)

res = comfyui_server.create()
print("ComfyUI hosted at:", res.url)
