from beam import endpoint, Image
from beam import Image, Volume, endpoint, Output

CACHE_PATH = "./models"
BEAM_OUTPUT_PATH = "/tmp/image_sdx_turbo.png"
BASE_MODEL = "stabilityai/sdxl-turbo"

image = Image(
    python_version="python3.10",
    python_packages=[
        "diffusers[torch]",
        "transformers",
        "pillow",
    ],
)


def load_models():
    from diffusers import AutoPipelineForText2Image
    import torch

    pipe = AutoPipelineForText2Image.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")

    return pipe


@endpoint(
    name="sdxl-turbo",
    image=image,
    on_start=load_models,
    keep_warm_seconds=60,
    cpu=16,
    memory="32Gi",
    gpu="A100-40",
    volumes=[Volume(name="models", mount_path=CACHE_PATH)],
)
def generate(context, prompt):


    pipe = context.on_start_value

    image = pipe(prompt=prompt, num_inference_steps=8, guidance_scale=0.0).images[0]

    print(f"Saved Image: {image}")

    # Save image file
    image.save(BEAM_OUTPUT_PATH)
    output = Output(path=BEAM_OUTPUT_PATH)
    output.save()
    # Retrieve pre-signed URL for output file
    url = output.public_url()
    print(url)

    return {"image": url}
