from beam import function, Volume, Image

VOLUME_PATH = "./instant-id"


@function(
    image=Image()
    .add_python_packages(
        ["huggingface_hub", "datasets", "huggingface_hub[hf-transfer]", "gdown"]
    )
    .add_commands(
        [
            "apt-get update",
            "apt-get install -y unzip",
        ]
    )
    .with_envs("HF_HUB_ENABLE_HF_TRANSFER=1"),
    memory="32Gi",
    cpu=4,
    secrets=["HF_TOKEN"],
    volumes=[Volume(name="instant-id", mount_path=VOLUME_PATH)],
)
def upload():
    from huggingface_hub import hf_hub_download, snapshot_download
    import gdown
    import os

    snapshot_download(
        repo_id="wangqixun/YamerMIX_v8",
        local_dir=f"{VOLUME_PATH}/weights",
    )

    controlnet_dir = f"{VOLUME_PATH}/checkpoints/ControlNetModel"
    os.makedirs(controlnet_dir, exist_ok=True)

    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ControlNetModel/config.json",
        local_dir=controlnet_dir,
    )
    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ControlNetModel/diffusion_pytorch_model.safetensors",
        local_dir=controlnet_dir,
    )

    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ip-adapter.bin",
        local_dir=f"{VOLUME_PATH}/checkpoints",
    )

    os.makedirs(f"{VOLUME_PATH}/models", exist_ok=True)
    gdown.download(
        url="https://drive.google.com/uc?id=18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8",
        output=f"{VOLUME_PATH}/models/antelopev2.zip",
        quiet=False,
        fuzzy=True,
    )
    os.system(f"unzip {VOLUME_PATH}/models/antelopev2.zip -d {VOLUME_PATH}/models/")


if __name__ == "__main__":
    upload()
