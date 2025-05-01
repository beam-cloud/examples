from beam import endpoint, Image, Output, Volume

VOLUME_PATH = "/instant-id"

image = (
    Image(python_version="python3.11")
    .add_commands(
        [
            "apt-get update",
            "apt-get install -y libgl1-mesa-glx libglib2.0-0 build-essential g++ python3-dev wget unzip",
        ]
    )
    .add_python_packages(
        [
            "opencv-python==4.9.0.80",
            "transformers==4.37.0",
            "accelerate==0.26.1",
            "insightface==0.7.3",
            "diffusers==0.25.1",
            "onnxruntime==1.16.3",
            "omegaconf==2.3.0",
            "gradio==3.50.2",
            "peft==0.8.2",
            "controlnet-aux==0.0.7",
            "huggingface_hub==0.25.2",
            "gdown",
        ]
    )
    .add_commands(
        [
            "git clone https://github.com/zsxkib/InstantID.git /instantid",
        ]
    )
)


@endpoint(
    name="instant-id",
    cpu=12,
    memory="32Gi",
    gpu="A10G",
    image=image,
    volumes=[Volume(name="instant-id", mount_path=VOLUME_PATH)],
)
def generate_image(
    image: str = "https://live-production.wcms.abc-cdn.net.au/a241657894f4d79f0c3ea0705f0f1f07?impolicy=wcms_crop_resize&cropH=1989&cropW=2992&xPos=8&yPos=8&width=862&height=575",
    prompt: str = "film noir style, ink sketch|vector, male man, highly detailed, sharp focus, ultra sharpness, monochrome, high contrast, dramatic shadows, 1940s style, mysterious, cinematic",
):
    import sys

    sys.path.append("/instantid")
    from diffusers.utils import load_image
    from diffusers.models import ControlNetModel
    import cv2
    import torch
    import numpy as np
    from insightface.app import FaceAnalysis
    import requests
    import uuid

    from pipeline_stable_diffusion_xl_instantid import (
        StableDiffusionXLInstantIDPipeline,
        draw_kps,
    )

    app = FaceAnalysis(
        name="antelopev2",
        root=VOLUME_PATH,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    face_adapter = f"{VOLUME_PATH}/checkpoints/ip-adapter.bin"
    controlnet_path = f"{VOLUME_PATH}/checkpoints/ControlNetModel/ControlNetModel"
    base_model = f"{VOLUME_PATH}/weights"

    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, torch_dtype=torch.float16
    )

    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        base_model, controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.cuda()

    pipe.load_ip_adapter_instantid(face_adapter)

    response = requests.get(image)
    response.raise_for_status()

    img_path = "/tmp/" + str(uuid.uuid4()) + ".png"
    with open(img_path, "wb") as f:
        f.write(response.content)

    face_image = load_image(img_path)

    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(
        face_info,
        key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
    )[-1]
    face_emb = face_info["embedding"]
    face_kps = draw_kps(face_image, face_info["kps"])

    image = pipe(
        prompt,
        image_embeds=face_emb,
        image=face_kps,
        controlnet_conditioning_scale=0.8,
        ip_adapter_scale=0.8,
    ).images[0]

    output_path = "/tmp/" + str(uuid.uuid4()) + ".png"
    image.save(output_path)
    output = Output(path=output_path)
    output.save()
    output_url = output.public_url()

    return {"output_url": output_url}
