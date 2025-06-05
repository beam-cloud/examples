from beam import function, Volume, Image
from transformers import AutoModelForCausalLM, AutoTokenizer

@function(
  app="volume-imports",
  name="huggingface-clone-model",
  secrets=["HUGGINGFACE_TOKEN"],
  memory="8gb",
  image=Image(
    python_packages=["torch","transformers"]
  ),
  volumes=[Volume(name="huggingface_models", mount_path="/huggingface_models")]
)
def handler(*, model_name: str = ""):
    if not model_name:
        raise ValueError("model_name is required")
    
    print(f"Downloading model: {model_name}")
    
    try:
        # Download model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Save to local volume
        save_path = f"/huggingface_models/{model_name.replace('/', '_')}"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved to: {save_path}")
        
        return {
            "model_name": model_name,
            "saved_path": save_path
        }
    except Exception as e:
        print(f"Failed to download model: {str(e)}")
        raise Exception(f"Failed to download model: {str(e)}")

if __name__ == "__main__":
    handler(model_name="distilbert/distilgpt2")