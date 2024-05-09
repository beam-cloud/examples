from beam import task_queue, Image, Volume, PythonVersion

device = "cuda"

image = Image(
    python_version=PythonVersion.Python310,
    python_packages=[
        "numpy",
        "git+https://github.com/openai/whisper.git",
        "pytube@git+https://github.com/felipeucelli/pytube@03d72641191ced9d92f31f94f38cfb18c76cfb05",
    ],
    commands=["apt-get update && apt-get install -y ffmpeg"],
)


def load_models():
    import whisper

    model = whisper.load_model("small", device=device, download_root="./cache")
    return model


@task_queue(
    image=image,
    on_start=load_models,
    cpu=1,
    memory="32Gi",
    gpu="T4",
    volumes=[
        Volume(mount_path="./cache", name="cache"),
    ],
)
def transcribe(context, video_url):
    from pytube import YouTube
    import os

    # Create YouTube object
    yt = YouTube(video_url)
    video = yt.streams.filter(only_audio=True).first()

    # Download audio to the output path
    out_file = video.download(output_path="./")
    base, ext = os.path.splitext(out_file)
    new_file = base + ".mp3"
    os.rename(out_file, new_file)
    a = new_file

    # Retrieve model from loader
    model = context.on_start_value
    # Inference
    result = model.transcribe(a)

    print(result["text"])
    return {"pred": result["text"]}
