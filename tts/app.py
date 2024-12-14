from beam import endpoint, env, Image, Output

if env.is_remote():
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    import soundfile as sf
    import uuid

def load_models():
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1").to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    return model, tokenizer

parlertts_image = (
    Image(
        base_image="nvidia/cuda:12.2.0-devel-ubuntu22.04",
        python_version="python3.10",
        python_packages=["torch", "transformers", "soundfile", "Pillow", "wheel", "packaging", "ninja"]
    )
    .add_commands(["apt update && apt install git -y", "pip install git+https://github.com/huggingface/parler-tts.git"])
)

@endpoint(
    name="parler-tts",
    on_start=load_models,
    cpu=2,
    memory="32Gi",
    gpu="A10G",
    gpu_count=2,
    image=parlertts_image
)
def generate_speech(context, **inputs):
    model, tokenizer = context.on_start_value

    prompt = inputs.pop("prompt", None)
    description = inputs.pop("description", None)

    if not prompt or not description:
        return {"error": "Please provide a prompt and description"}
    
    device = "cuda:0"

    input_ids = tokenizer(
        description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(
        prompt, return_tensors="pt").input_ids.to(device)

    generation = model.generate(
        input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()

    file_name = f"/tmp/parler_tts_out_{uuid.uuid4()}.wav"

    sf.write(file_name, audio_arr, model.config.sampling_rate)
   
    output_file = Output(path=file_name)
    output_file.save()
    public_url = output_file.public_url(expires=400)
    print(public_url)
    return {"output_url": public_url}