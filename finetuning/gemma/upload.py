from beam import function, Volume, Image, env

if env.is_remote():
    from huggingface_hub import snapshot_download
    from datasets import load_dataset

VOLUME_PATH = "./gemma-ft"

@function(
    image=Image(
        python_packages=[
            "huggingface_hub",
            "datasets"
            "huggingface_hub[hf-transfer]"
        ],
    ).with_envs("HF_HUB_ENABLE_HF_TRANSFER=1"),
    memory="32Gi",
    cpu=4,
    secrets=["HF_TOKEN"],
    volumes=[Volume(name="gemma-ft", mount_path=VOLUME_PATH)],
)
def upload():
    snapshot_download(
        repo_id="google/gemma-2b",
        local_dir=f"{VOLUME_PATH}/weights"
    )

    dataset = load_dataset("OpenAssistant/oasst1", split="train")
    dataset.save_to_disk(f"{VOLUME_PATH}/data")
    print("Files uploaded successfully")


if __name__ == "__main__":
    upload()
