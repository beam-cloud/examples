from beam import endpoint, Image, Volume, env
import base64
import requests
from tempfile import NamedTemporaryFile


BEAM_VOLUME_PATH = "./cached_models"


# These packages will be installed in the remote container
if env.is_remote():
    from faster_whisper import WhisperModel, download_model


# This runs once when the container first starts
def load_models():
    model_path = download_model("large-v3", cache_dir=BEAM_VOLUME_PATH)
    model = WhisperModel(model_path, device="cuda", compute_type="float16")
    return model


@endpoint(
    on_start=load_models,
    name="faster-whisper",
    cpu=2,
    memory="32Gi",
    gpu="A10G",
    image=Image(
        base_image="nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04",
        python_version="python3.10",
        python_packages=["git+https://github.com/SYSTRAN/faster-whisper.git" "huggingface_hub[hf-transfer]"],
    ).with_envs("HF_HUB_ENABLE_HF_TRANSFER=1"),
    volumes=[
        Volume(
            name="cached_models",
            mount_path=BEAM_VOLUME_PATH,
        )
    ],
)
def transcribe(context, **inputs):
    # Retrieve cached model from on_start
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

            segments, _ = model.transcribe(
                temp.name, beam_size=5, language=language)

            for segment in segments:
                text += segment.text + " "

            print(text)
            return {"text": text}

        except Exception as e:
            return {"error": f"Something went wrong: {e}"}
