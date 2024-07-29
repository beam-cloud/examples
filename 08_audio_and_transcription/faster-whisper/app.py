from beam import endpoint, Image, Volume, Output, env
import base64
import requests
from tempfile import NamedTemporaryFile

if env.is_remote():
    from faster_whisper import WhisperModel, download_model

volume_path = "./cached_models"


def load_models():
    model_path = download_model("large-v3", cache_dir=volume_path)
    model = WhisperModel(model_path, device="cuda", compute_type="float16")
    return model


@endpoint(
    on_start=load_models,
    name="faster-whisper",
    cpu=1,
    memory="16Gi",
    gpu="T4",
    image=Image(
        python_version="python3.9",
        base_image="docker.io/nvidia/cuda:12.0-cudnn8-devel-ubuntu22.04",
        python_packages=["torch==2.1.0", "faster-whisper==0.10.0"],
    ),
    volumes=[
        Volume(
            name="cached_models",
            mount_path=volume_path,
        )
    ],
)
def transcribe(context, **inputs):
    # Retrieve model from on_start
    model = context.on_start_value

    # Inputs passed to API
    language = inputs.get("language")
    audio_base64 = inputs.get("audio_file")
    url = inputs.get("url")

    if audio_base64 and url:
        return {"error": "Only a base64 audio file OR a URL can be passed to the API."}
    if not audio_base64 and not url:
        return {
            "error": "Please provide either an audio file in base64 string format or a URL."
        }

    binary_data = None

    if audio_base64:
        binary_data = base64.b64decode(audio_base64.encode("utf-8"))
    elif url:
        resp = requests.get(url)
        binary_data = resp.content

    text = ""

    with NamedTemporaryFile() as temp:
        try:
            # Write the audio data to the temporary file
            temp.write(binary_data)
            temp.flush()

            segments, _ = model.transcribe(temp.name, beam_size=5, language=language)

            for segment in segments:
                text += segment.text + " "

            print(text)
            with open("output.txt", "w") as f:
                f.write(text)

            return {"text": text}

        except Exception as e:
            return {"error": f"Something went wrong: {e}"}
