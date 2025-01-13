from beam import endpoint, env, Volume, Image, Output

VOLUME_PATH = "./mochi-1-preview"

if env.is_remote():
    import torch
    from diffusers import MochiPipeline
    from diffusers.utils import export_to_video
    import uuid
 
def load_models():
    pipe = MochiPipeline.from_pretrained(
        f"{VOLUME_PATH}/weights", variant="bf16", torch_dtype=torch.bfloat16)
    return pipe 


mochi_image = (
    Image(
        python_version="python3.11",
        python_packages=["torch", "transformers", "accelerate",
                         "sentencepiece", "imageio-ffmpeg", "imageio", "ninja", "huggingface_hub[hf-transfer]"]
    )
    .add_commands(["apt update && apt install git -y", "pip install git+https://github.com/huggingface/diffusers.git"])
    .with_envs("HF_HUB_ENABLE_HF_TRANSFER=1")
)

@endpoint(
    name="mochi-1-preview",
    on_start=load_models,
    cpu=4,
    memory="32Gi",
    gpu="A10G",
    gpu_count=2,
    image=mochi_image,
    volumes=[Volume(name="mochi-1-preview", mount_path=VOLUME_PATH)],
    timeout=-1
)
def generate_video(context, **inputs):
    pipe = context.on_start_value

    prompt = inputs.pop("prompt", None)

    if not prompt:
        return {"error": "Please provide a prompt"}

    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    frames = pipe(prompt, num_frames=40).frames[0]

    file_name = f"/tmp/mochi_out_{uuid.uuid4()}.mp4"
    
    export_to_video(frames, file_name, fps=15)

    output_file = Output(path=file_name)
    output_file.save()
    public_url = output_file.public_url(expires=-1)
    print(public_url)
    return {"output_url": public_url}
