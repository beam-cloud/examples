"""
*** Whisper Example *** 

This app deploys a serverless GPU function which takes a audio URL as input ("audio_url") 
and transcribes the audio provided using Whisper.

Deploy this by running:

`beam deploy app.py:transcibe`
"""

from beam import endpoint, Image, Volume

device = "cuda"

image = (
    Image(python_version="python3.10")
    .add_commands(["apt-get update && apt-get install -y ffmpeg"])
    .add_python_packages(
        [
            "numpy",
            "git+https://github.com/openai/whisper.git",
            "huggingface_hub[hf-transfer]",
        ]
    )
    .with_envs("HF_HUB_ENABLE_HF_TRANSFER=1")
)


# This runs when the container first starts and is used to cache the model on disk
def load_models():
    import whisper

    model = whisper.load_model("small", device=device, download_root="./cache")
    return model


@endpoint(
    name="whisper",
    image=image,
    on_start=load_models,
    cpu=1,
    memory="32Gi",
    gpu="T4",
    volumes=[
        Volume(
            mount_path="./cache", name="cache"
        ),  # The downloaded model is stored here
        Volume(mount_path="./audios", name="audios"),  # Audio is stored here
    ],
)
# Inference Function. Context is the value passed down from the loader, audio_url is the API input
def transcribe(context, **inputs):
    model = context.on_start_value

    url = inputs.get("audio_url")

    result = model.transcribe(url)

    print(result["text"])

    return {"text": result["text"]}
