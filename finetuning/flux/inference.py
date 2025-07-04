from beam import function, Volume, Output, Image
import os
import base64
from io import BytesIO

VOLUME_PATH = "./flux-lora-data"

def get_pipeline(lora_name: str = None):
    """
    Loads the base FLUX pipeline and optionally attaches a LoRA adapter.
    The base model is cached on the persistent volume to speed up subsequent loads.
    """
    print("Loading FLUX pipeline...")
    
    from diffusers import FluxPipeline
    import torch
    
    cache_path = os.path.join(VOLUME_PATH, "hf_cache")
    os.makedirs(cache_path, exist_ok=True)
    
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.float16,
        token=os.getenv("HF_TOKEN"),
        cache_dir=cache_path
    ).to("cuda")
    
    if lora_name:
        lora_folder_name = f"flux_lora_{lora_name}"
        lora_filename = f"{lora_folder_name}.safetensors"
        lora_full_path = os.path.join(VOLUME_PATH, lora_folder_name, lora_filename)

        if os.path.exists(lora_full_path):
            print(f"Loading LoRA adapter: {lora_filename}")
            try:
                lora_dir = os.path.dirname(lora_full_path)
                pipeline.load_lora_weights(lora_dir, weight_name=lora_filename)
                print(f"Successfully attached LoRA: {lora_name}")
                return pipeline, lora_full_path
            except Exception as e:
                print(f"Failed to load LoRA adapter '{lora_filename}': {e}")
        else:
            print(f"LoRA adapter not found for '{lora_name}'. Using the base model.")
    else:
        print("No LoRA name provided, using the base model.")
        
    return pipeline, None

@function(
    name="simple-flux-generate",
    gpu="A100-40",
    cpu=4,
    memory="32Gi",
    volumes=[Volume(name="flux-lora-data", mount_path=VOLUME_PATH)],
    image=Image(python_version="python3.12")
        .add_commands([
            "apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1",
            "git clone https://github.com/ostris/ai-toolkit.git /ai-toolkit",
            "cd /ai-toolkit && git submodule update --init --recursive",
            "pip3.12 install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126",
            "pip3.12 install pyyaml requests pillow opencv-python-headless",
            "cd /ai-toolkit && pip3.12 install -r requirements.txt"
        ]),
    secrets=["HF_TOKEN"]
)
def generate_image(
    prompt: str,
    lora_name: str = None,
    width: int = 1024,
    height: int = 1024,
    steps: int = 35,
    guidance: float = 3.0,
    seed: int = None,
    lora_scale: float = 0.9
):
    """
    Generate image with a dynamically loaded LoRA adapter.
    """
    pipeline, lora_path = get_pipeline(lora_name=lora_name)
    
    import torch
    
    print(f"Generating: '{prompt}'")
    print(f"{width}x{height}, steps: {steps}, guidance: {guidance}, lora: {lora_name or 'None'}, scale: {lora_scale}")
    
    generator = None
    if seed:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        print(f"Using seed: {seed}")

    with torch.no_grad():
        result = pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator
        )
    
    image = result.images[0]
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    filename = f"generated_{seed or 'random'}_{hash(prompt) % 10000}.png"
    output = Output.from_pil_image(image)
    output.save()
    
    local_path = os.path.join(VOLUME_PATH, filename)
    image.save(local_path)
    
    print(f"Generated and saved: {filename}")
    print(f"Local path: {local_path}")
    print(f"Public URL: {output.public_url}")
    
    return {
        "status": "success",
        "image": img_base64,
        "url": output.public_url,
        "prompt": prompt,
        "settings": {
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
            "lora": lora_name
        }
    }