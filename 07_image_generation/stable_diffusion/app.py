"""
** Stable Diffusion on Beam ** 

The code below shows how to deploy a serverless inference API for running stable diffusion.
"""

from beam import Image, Volume, task_queue, Output


CACHE_PATH = "./models"
model_id = "runwayml/stable-diffusion-v1-5"

# The environment your app runs on
image = Image(
    python_version="python3.9",
    python_packages=[
        "diffusers[torch]>=0.10",
        "transformers",
        "torch",
        "pillow",
        "accelerate",
        "safetensors",
        "xformers",
    ],
)


# This runs once when the container first boots
def load_models():
    import torch
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        variant="fp16",
        torch_dtype=torch.float16,
        cache_dir=CACHE_PATH,
    ).to("cuda")

    return pipe


@task_queue(
    image=image,
    on_start=load_models,
    keep_warm_seconds=60,
    cpu=2,
    memory="32Gi",
    gpu="A10G",
    # Mount two storage volumes to the app: one for caching model weights, and one for the generated images
    volumes=[
        Volume(name="models", mount_path=CACHE_PATH),
    ],
)
def generate(context, prompt):
    import torch

    # Retrieve pre-loaded model from loader
    pipe = context.on_start_value

    torch.backends.cuda.matmul.allow_tf32 = True

    with torch.inference_mode():
        with torch.autocast("cuda"):
            image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

    # Save image file
     Output.from_pil_image(image)
    output.save()

    # Retrieve pre-signed URL for output file
    url = output.public_url(expires=400)
    print(url)

    return {"image": url}
