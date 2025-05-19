from beam import Image, asgi, Output
import requests
from pathlib import Path

image = (
    Image()
    .add_commands(["apt update && apt install git -y"])
    .add_python_packages(
        [
            "fastapi[standard]==0.115.4",
            "comfy-cli",
            "huggingface_hub[hf_transfer]==0.26.2",
        ]
    )
    .add_commands(
        [
            "yes | comfy install --nvidia --version 0.3.10",
            "comfy node install was-node-suite-comfyui@1.0.2",
            "mkdir -p /root/comfy/ComfyUI/models/checkpoints/",
            "huggingface-cli download Comfy-Org/flux1-schnell flux1-schnell-fp8.safetensors --cache-dir /comfy-cache",
            "ln -s /comfy-cache/models--Comfy-Org--flux1-schnell/snapshots/f2808ab17fe9ff81dcf89ed0301cf644c281be0a/flux1-schnell-fp8.safetensors /root/comfy/ComfyUI/models/checkpoints/flux1-schnell-fp8.safetensors",
        ]
    )
)


def download_image_from_url(url: str, filename: str = "input_image.png"):
    target_path = Path("/root/comfy/ComfyUI/input") / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download image: {response.status_code}")

    with open(target_path, "wb") as f:
        f.write(response.content)

    return target_path


def hf_download():
    import subprocess
    from huggingface_hub import hf_hub_download

    dream_model = hf_hub_download(
        repo_id="Bruhn/Lab_merge",
        filename="dreamCreationVirtual3DECommerce_v10.safetensors",
        cache_dir="/comfy-cache",
    )

    vae_model = hf_hub_download(
        repo_id="stabilityai/sd-vae-ft-mse-original",
        filename="vae-ft-mse-840000-ema-pruned.safetensors",
        cache_dir="/comfy-cache",
    )

    controlnet_model = hf_hub_download(
        repo_id="comfyanonymous/ControlNet-v1-1_fp16_safetensors",
        filename="control_v11p_sd15_scribble_fp16.safetensors",
        cache_dir="/comfy-cache",
    )

    subprocess.run(
        "mkdir -p /root/comfy/ComfyUI/models/checkpoints/",
        shell=True,
        check=True,
    )
    subprocess.run(
        "mkdir -p /root/comfy/ComfyUI/models/vae/",
        shell=True,
        check=True,
    )
    subprocess.run(
        "mkdir -p /root/comfy/ComfyUI/models/controlnet/",
        shell=True,
        check=True,
    )

    subprocess.run(
        f"ln -s {dream_model} /root/comfy/ComfyUI/models/checkpoints/dreamCreationVirtual3DECommerce_v10.safetensors",
        shell=True,
        check=True,
    )
    subprocess.run(
        f"ln -s {vae_model} /root/comfy/ComfyUI/models/vae/vae-ft-mse-840000-ema-pruned.safetensors",
        shell=True,
        check=True,
    )
    subprocess.run(
        f"ln -s {controlnet_model} /root/comfy/ComfyUI/models/controlnet/control_v11p_sd15_scribble_fp16.safetensors",
        shell=True,
        check=True,
    )

    cmd = "comfy launch --background"
    subprocess.run(cmd, shell=True, check=True)


@asgi(
    name="comfy",
    image=image,
    on_start=hf_download,
    cpu=8,
    memory="32Gi",
    gpu="A100-40",
    timeout=-1,
)
def handler():
    from fastapi import FastAPI, HTTPException
    import subprocess
    import json
    from pathlib import Path
    import uuid
    from typing import Dict

    app = FastAPI()

    # This is where you specify the path to your workflow file.
    # Make sure "workflow_api.json" exists in the same directory as this script.
    WORKFLOW_FILE = Path(__file__).parent / "workflow_api.json"
    OUTPUT_DIR = Path("/root/comfy/ComfyUI/output")

    @app.post("/generate")
    async def generate(item: Dict):
        if not WORKFLOW_FILE.exists():
            raise HTTPException(status_code=500, detail="Workflow file not found.")

        image_url = item.get("image_url")
        prompt = item.get("prompt")

        if not image_url or not prompt:
            raise HTTPException(status_code=400, detail="Missing image_url or prompt")

        downloaded_image_path = download_image_from_url(image_url, "input.jpg")
        request_id = uuid.uuid4().hex

        workflow_data = json.loads(WORKFLOW_FILE.read_text())
        workflow_data["6"]["inputs"]["text"] = item["prompt"]
        workflow_data["11"]["inputs"]["image"] = str(downloaded_image_path)

        new_workflow_file = Path(f"{request_id}.json")
        new_workflow_file.write_text(json.dumps(workflow_data, indent=4))

        # Run inference
        cmd = (
            f"comfy run --workflow {new_workflow_file} --wait --timeout 1200 --verbose"
        )
        subprocess.run(cmd, shell=True, check=True)

        image_files = list(OUTPUT_DIR.glob("*"))

        # Find the latest image
        latest_image = max(
            (f for f in image_files if f.suffix.lower() in {".png", ".jpg", ".jpeg"}),
            key=lambda f: f.stat().st_mtime,
            default=None,
        )

        if not latest_image:
            raise HTTPException(status_code=404, detail="No output image found.")

        output_file = Output(path=latest_image)
        output_file.save()
        public_url = output_file.public_url(expires=-1)
        print(public_url)
        return {"output_url": public_url}

    return app
