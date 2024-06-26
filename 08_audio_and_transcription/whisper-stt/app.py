"""
*** Whisper Example *** 

This app deploys a serverless GPU function which takes a Youtube URL as input ("video_url") 
and transcribes the video provided using Whisper.

Deploy this by running:

`beam deploy app.py:transcibe`
"""

from beam import endpoint, Image, Volume, PythonVersion

device = "cuda"

image = Image(
    python_version=PythonVersion.Python38,
    python_packages=[
        "numpy",
        "git+https://github.com/openai/whisper.git",
        "pytube@git+https://github.com/felipeucelli/pytube@03d72641191ced9d92f31f94f38cfb18c76cfb05",
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
    from pytube import YouTube
    import os

    # Create YouTube object
    yt = YouTube(video_url)
    video = yt.streams.filter(only_audio=True).first()

    # Download audio to the `videos` volume
    out_file = video.download(output_path="./videos")

    base, ext = os.path.splitext(out_file)
    new_file = base + ".mp3"
    os.rename(out_file, new_file)
    a = new_file

    # Retrieve model from loader
    model = context.on_start_value
    # Inference
    result = model.transcribe(a)

    print(result["text"])

    return {"text": result["text"]}
