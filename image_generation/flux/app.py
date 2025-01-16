"""
** Flux on Beam ** 

The code below shows how to deploy a serverless inference API for running flux.
"""

from beam import Image, Volume, endpoint, Output

# The model is cached in a Beam Storage Volume at this path
CACHE_PATH = "./models"

# The container image for running Flux
image = (
    Image(python_version="python3.9")
    .add_python_packages(
        [
            "diffusers[torch]>=0.10",
            "transformers",
            "torch",
            "pillow",
            "accelerate",
            "sentencepiece",
            "protobuf",
            "safetensors",
            "xformers",
            "huggingface_hub[hf-transfer]",
        ],
    )
    .with_envs("HF_HUB_ENABLE_HF_TRANSFER=1")
)


# This runs once when the container first boots
def load_models():
    import torch
    from diffusers import FluxPipeline

    # Load model
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, cache_dir=CACHE_PATH
    )
    # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    pipe.enable_model_cpu_offload()

    return pipe


@endpoint(
    name="serverless-flux",
    secrets=[
        "HF_TOKEN"
    ],  # Make sure you've saved your HF_TOKEN to Beam: `beam secret create HF_TOKEN [value]`
    image=image,
    on_start=load_models,
    keep_warm_seconds=60,
    cpu=2,
    memory="32Gi",
    gpu="A100-40",
    volumes=[
        Volume(name="models", mount_path=CACHE_PATH)
    ],  # Cached model is stored here
)
def generate(context, **inputs):
    import torch

    # Retrieve pre-loaded model from loader
    pipe = context.on_start_value

    prompt = inputs.get("prompt", "a penguin riding the subway")
    if not prompt:
        return {"error": "Please provide prompts for image generation."}

    # Generate image
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]

    # Save image file
    output = Output.from_pil_image(image)
    output.save()

    # Retrieve pre-signed URL for output file
    url = output.public_url(expires=400)
    print(url)

    return {"image": url}
