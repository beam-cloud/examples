from beam import Image, endpoint, Output, env

if env.is_remote():
    import torchaudio
    from zonos.model import Zonos
    from zonos.conditioning import make_cond_dict
    from zonos.utils import DEFAULT_DEVICE as device
    import os
    import uuid

image = (
    Image(
        base_image="nvidia/cuda:12.4.1-devel-ubuntu22.04", python_version="python3.11"
    )
    .add_commands(["apt update && apt install -y espeak-ng git"])
    .add_commands(
        [
            "pip install -U uv",
            "git clone https://github.com/Zyphra/Zonos.git /tmp/Zonos",
            "cd /tmp/Zonos && pip install setuptools wheel && pip install -e .",
        ]
    )
)


@endpoint(name="zonos", image=image, cpu=12, memory="32Gi", gpu="A100-40", timeout=-1)
def generate(**inputs):
    text = inputs.pop("text", None)

    if not text:
        return {"error": "Please provide a text"}

    os.chdir("/tmp/Zonos")

    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

    wav, sampling_rate = torchaudio.load("assets/exampleaudio.mp3")
    speaker = model.make_speaker_embedding(wav, sampling_rate)

    cond_dict = make_cond_dict(text=text, speaker=speaker, language="en-us")
    conditioning = model.prepare_conditioning(cond_dict)

    codes = model.generate(conditioning)

    file_name = f"/tmp/zonos_out_{uuid.uuid4()}.wav"

    wavs = model.autoencoder.decode(codes).cpu()
    torchaudio.save(file_name, wavs[0], model.autoencoder.sampling_rate)
    output_file = Output(path=file_name)
    output_file.save()
    public_url = output_file.public_url(expires=1200000000)
    print(public_url)
    return {"output_url": public_url}
