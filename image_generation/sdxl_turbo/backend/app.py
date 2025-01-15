from beam import Image, Volume, endpoint, Output

CACHE_PATH = "./models"
BASE_MODEL = "stabilityai/sdxl-turbo"

image = Image(
    python_version="python3.10",
    python_packages=[
        "diffusers[torch]",
        "transformers",
        "pillow",
        "huggingface_hub[hf-transfer]",
    ],
).with_envs("HF_HUB_ENABLE_HF_TRANSFER=1")


# This runs once when the container first starts
def load_models():
    from diffusers import AutoPipelineForText2Image
    import torch

    pipe = AutoPipelineForText2Image.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.to("cuda")

    return pipe


@endpoint(
    name="sdxl-turbo",
    image=image,
    on_start=load_models,
    keep_warm_seconds=60,
    cpu=2,
    memory="20Gi",
    gpu="A10G",
    volumes=[Volume(name="models", mount_path=CACHE_PATH)],
)
def generate(context, prompt):
    # Retrieve cached model from on_start function
    pipe = context.on_start_value

    # Inference
    image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]

    # Save image file
    output = Output.from_pil_image(image)
    output.save()

    # Retrieve pre-signed URL for output file
    url = output.public_url(expires=400)
    print(url)

    return {"image": url}
