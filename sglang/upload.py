from beam import function, Volume, Image, env

if env.is_remote():
    from huggingface_hub import snapshot_download

VOLUME_PATH = "./llama3.2",

@function(
    image=Image(
        python_packages=[
            "huggingface_hub",
        ]
    ),
    memory="32Gi",
    cpu=4,
    secrets=["HF_TOKEN"],
    volumes=[Volume(name="mochi-1-preview", mount_path=VOLUME_PATH)]
)
def upload():
    snapshot_download(
        repo_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        local_dir=f"{VOLUME_PATH}/weights"
    )
    
    print("Files uploaded successfully")

if __name__ == "__main__":
    upload()