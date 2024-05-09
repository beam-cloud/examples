"""
** Stable Diffusion on Beam ** 

The code below shows how to deploy a serverless inference API for running stable diffusion.
"""

from beam import Image, Volume, task_queue


CACHE_PATH = "./models"
IMAGES_PATH = "./images"
model_id = "runwayml/stable-diffusion-v1-5"

# The environment your app runs on
image = Image(
    python_version="python3.8",
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
    memory="16Gi",
    gpu="A10G",
    # Mount two storage volumes to the app: one for caching model weights, and one for the generated images
    volumes=[
        Volume(name="models", mount_path=CACHE_PATH),
        Volume(name="images", mount_path=IMAGES_PATH),
    ],
)
def generate(context, prompt):
    import os
    import torch

    # Retrieve pre-loaded model from loader
    pipe = context.on_start_value

    torch.backends.cuda.matmul.allow_tf32 = True

    with torch.inference_mode():
        with torch.autocast("cuda"):
            image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

    print(f"Saved Image: {image}")

    # Save image to Beam Volume
    image_path = os.path.join(IMAGES_PATH, f"{prompt.replace(' ', '_')}.png")
    image.save(image_path)

    # Saved image can be accessed in the 'volumes' page of your dashboard:
    # https://platform.beam.cloud/volumes/
    # Or, in the CLI by running `beam ls images`
    print(f"Image saved to: {image_path}")
