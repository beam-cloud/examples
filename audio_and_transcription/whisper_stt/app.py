"""
*** Whisper Example *** 

This app deploys a serverless GPU function which takes a Youtube URL as input ("video_url") 
and transcribes the video provided using Whisper.

Deploy this by running:

`beam deploy app.py:transcibe`
"""

from beam import endpoint, Image, Volume

device = "cuda"

image = Image(
    python_version="python3.10",
    python_packages=[
        "numpy",
        "git+https://github.com/openai/whisper.git",
        "yt-dlp"
    ],
    commands=["apt-get update && apt-get install -y ffmpeg"],
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
        Volume(mount_path="./videos", name="videos"),  # Video audio is stored here
    ],
)
# Inference Function. Context is the value passed down from the loader, video_url is the API input
def transcribe(context, video_url):
    import uuid
    import subprocess
    output_path = f"./videos/{uuid.uuid4()}.mp3"

    subprocess.run(["yt-dlp", "-x", "--audio-format", "mp3", "-o", output_path, video_url], check=True)

    model = context.on_start_value
    result = model.transcribe(output_path)

    print(result["text"])

    return {"text": result["text"]}