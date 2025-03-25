"""
This is a ComfyUI server that allows you to generate images using the Flux1 Schnell model.

To deploy this app, run "python app.py" in your terminal.
"""

from beam import Image, Pod

image = (
    Image()
    .add_commands(["apt update && apt install git -y"])
    .add_python_packages(
        [
            "fastapi[standard]==0.115.4",
            "comfy-cli==1.3.5",
            "huggingface_hub[hf_transfer]==0.26.2",
        ]
    )
    .add_commands(
        [
            "comfy --skip-prompt install --nvidia --version 0.3.10",
            "comfy node install was-node-suite-comfyui@1.0.2",
            "mkdir -p /root/comfy/ComfyUI/models/checkpoints/",
            "huggingface-cli download Comfy-Org/flux1-schnell flux1-schnell-fp8.safetensors --cache-dir /comfy-cache",
            "ln -s /comfy-cache/models--Comfy-Org--flux1-schnell/snapshots/f2808ab17fe9ff81dcf89ed0301cf644c281be0a/flux1-schnell-fp8.safetensors /root/comfy/ComfyUI/models/checkpoints/flux1-schnell-fp8.safetensors",
        ]
    )
)

comfyui_server = Pod(
    image=image,
    ports=[8000],
    cpu=12,
    memory="32Gi",
    gpu="A100-40",
    entrypoint=["sh", "-c", "comfy launch -- --listen 0.0.0.0 --port 8000"],
)

res = comfyui_server.create()
print("✨ ComfyUI hosted at:", res.url)
