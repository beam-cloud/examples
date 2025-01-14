from beam import endpoint, Image, Volume, env

if env.is_remote():
    import torch
    import whisperx
    import gc


# Define the custom image
image = (
    Image()
    .add_commands(["apt-get update -y", "apt-get install ffmpeg -y"])
    .add_python_packages(
        ["faster-whisper==1.0.1", "whisperx==3.1.5", "torchaudio==2.0.2", "huggingface_hub[hf-transfer]"]
    ).with_envs("HF_HUB_ENABLE_HF_TRANSFER=1")
)


volume_path = "./cached_models"
device = "cuda"
compute_type = "float16"
language_code = "en"


def on_start():
    model_name = "large-v2"

    # Load the main WhisperX model
    model = whisperx.load_model(
        model_name, device, download_root=volume_path, language=language_code
    )

    # Load the alignment model for word-level timestamps
    alignment_model, metadata = whisperx.load_align_model(
        language_code=language_code, device=device
    )

    return model, alignment_model, metadata


@endpoint(
    name="whisperx-deployment",
    image=image,
    cpu=4,
    memory="32Gi",
    gpu="A10G",
    volumes=[
        Volume(
            name="cached_models",
            mount_path=volume_path,
        )
    ],
    on_start=on_start,
)
def transcribe_audio(context, **inputs):
    # Retrieve values from on_start
    model, alignment_model, metadata = context.on_start_value

    url = inputs.get(
        "url",
        "https://audio-samples.github.io/samples/mp3/blizzard_unconditional/sample-0.mp3",
    )

    print(f"🚧 Loading audio from {url}...")
    audio = whisperx.load_audio(url)
    print("✅ Audio loaded")

    print("Transcribing...")
    result = model.transcribe(audio, batch_size=16)
    print("🎉 Transcription done:")
    print(result["segments"])

    # Delete model if low on GPU resources
    gc.collect()
    torch.cuda.empty_cache()
    del model

    print("Aligning...")
    result = whisperx.align(
        result["segments"],
        alignment_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    print("🎉 Alignment done")

    return {"result": result}
