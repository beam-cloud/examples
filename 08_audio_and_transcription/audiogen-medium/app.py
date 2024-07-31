"""
### AudioGen Medium ###

AudioGen is a simple and controllable model for audio generation developed
by Facebook AI Research.
"""

from beam import endpoint, Image, Volume, Output, env

if env.is_remote():
    import torch
    import base64
    import tempfile
    from audiocraft.data.audio import audio_write
    from audiocraft.models import AudioGen

MODEL_SIZE = "medium"
BEAM_VOLUME_PATH = "./cached_models"


def load_models():
    model = AudioGen.get_pretrained(f"facebook/audiogen-{MODEL_SIZE}", device="cuda")
    return model


@endpoint(
    on_start=load_models,
    name="audiogen-medium",
    cpu=2,
    memory="20Gi",
    gpu="A10G",
    image=Image(
        commands=["apt-get update -y && apt-get install ffmpeg -y"],
        python_version="python3.9",
        python_packages=[
            "torch",
            "git+https://github.com/facebookresearch/audiocraft.git",
            "torchaudio",
        ],
    ),
    volumes=[
        Volume(
            name="cached_models",
            mount_path=BEAM_VOLUME_PATH,
        )
    ],
)
def generate(context, **inputs):
    # Retrieve model from on_start
    model = context.on_start_value

    try:
        # Inputs passed to API
        prompts = inputs.pop("prompts", ["dog barking", "subway train"])
        if not prompts:
            return {"error": "Please provide prompts for audio generation."}

        duration = inputs.pop("duration", 8)
        model.set_generation_params(duration=duration)
        wav = model.generate(prompts)
        output_urls = []
        for idx, one_wav in enumerate(wav):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                audio_write(
                    tmpfile.name,
                    one_wav.cpu(),
                    model.sample_rate,
                    strategy="loudness",
                    loudness_compressor=True,
                )
                output = Output(path=f"{tmpfile.name}.wav")
                output.save()
                output_urls.append(output.public_url())

        return {"data": output_urls}
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
        }
