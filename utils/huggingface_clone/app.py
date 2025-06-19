from beam import function, Volume, Image
from huggingface_hub import snapshot_download
import os

@function(
  app="volume-imports",
  name="huggingface-clone-model",
  secrets=["HUGGINGFACE_TOKEN", "HF_TOKEN"],
  memory="8gb",
  gpu="T4",
  image=Image(
    python_packages=["torch", "huggingface_hub"]
  ),
  volumes=[Volume(name="huggingface_models", mount_path="/huggingface_models")]
)
def handler(*, model_name: str = ""):
    if not model_name:
        raise ValueError("model_name is required")
    
    print(f"Downloading model: {model_name}")
    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    
    try:
        os.makedirs(f"/huggingface_models/{model_name}", exist_ok=True)
        
        path = snapshot_download(repo_id=model_name, local_dir=f"/huggingface_models/{model_name}", token=token)
        print(f"Model downloaded to: {path}")
        
        return {
            "model_name": model_name,
            "saved_path": path
        }
    except Exception as e:
        print(f"Failed to download model: {str(e)}")
        raise Exception(f"Failed to download model: {str(e)}")

if __name__ == "__main__":
    handler(model_name="tencent/Hunyuan3D-2.1")