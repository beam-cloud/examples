from beam import endpoint, Volume, Image, QueueDepthAutoscaler
import torch
import os
import shutil
import yaml
import zipfile
import tempfile
import subprocess
import requests
from PIL import Image as PILImage

VOLUME_PATH = "./flux-lora-clean"

@endpoint(
    name="train-lora",
    gpu="H100",
    cpu=8,
    memory="32Gi",
    timeout=3600,
    keep_warm_seconds=60,
    volumes=[Volume(name="flux-lora-clean", mount_path=VOLUME_PATH)],
    image=Image(python_version="python3.11")
        .add_python_packages([
            "torch==2.6.0",
            "torchvision==0.21.0",
            "torchao==0.9.0",
            "safetensors",
            "transformers==4.52.4",
            "lycoris-lora==1.8.3",
            "flatten_json",
            "pyyaml",
            "oyaml",
            "tensorboard",
            "kornia",
            "invisible-watermark",
            "einops",
            "accelerate",
            "toml",
            "albumentations==1.4.15",
            "albucore==0.0.16",
            "pydantic",
            "omegaconf",
            "k-diffusion",
            "open_clip_torch",
            "timm",
            "prodigyopt",
            "controlnet_aux==0.0.10",
            "python-dotenv",
            "bitsandbytes",
            "hf_transfer",
            "lpips",
            "pytorch_fid",
            "optimum-quanto==0.2.4",
            "sentencepiece",
            "huggingface_hub",
            "peft",
            "gradio",
            "python-slugify",
            "opencv-python-headless",
            "pytorch-wavelets==1.3.0",
            "matplotlib==3.10.1",
            "diffusers",
            "packaging",
            "setuptools<70.0.0",
            "requests",
            "pillow"
        ])
        .add_commands([
            "apt-get update && apt-get install -y git libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1",
            "pip install git+https://github.com/jaretburkett/easy_dwpose.git",
            "pip install git+https://github.com/huggingface/diffusers@363d1ab7e24c5ed6c190abb00df66d9edb74383b"
        ])
        .with_envs("HF_HUB_ENABLE_HF_TRANSFER=1"),
    secrets=["HF_TOKEN"],
    autoscaler=QueueDepthAutoscaler(max_containers=3, tasks_per_container=1)
)
def train_lora(
    image_zip: str = None,
    trigger_word: str = "TOK",
    steps: int = 1000,
    learning_rate: float = 4e-4,
    rank: int = 32,
    alpha: int = 32,
    resolution: int = 1024
):
    """
    Fine-tune FLUX model with LoRA using uploaded dataset
    
    Args:
        image_zip: URL to zip file containing training images
        trigger_word: Token to associate with your concept  
        steps: Number of training steps
        learning_rate: Learning rate for training
        rank: LoRA rank (higher = more capacity)
        alpha: LoRA alpha (scaling factor)
        resolution: Training image resolution
    """
    print(f"Starting LoRA fine-tuning for '{trigger_word}'")
    print(f"Training: {steps} steps at {resolution}x{resolution}")
    
    # Setup environment
    setup_environment()
    
    # Process dataset
    image_count = process_dataset(image_zip, trigger_word, resolution)
    
    if image_count == 0:
        return {"error": "No training images found"}
    
    # Configure training
    config = create_training_config(
        trigger_word=trigger_word,
        steps=steps,
        learning_rate=learning_rate,
        rank=rank,
        alpha=alpha,
        resolution=resolution
    )
    
    # Run training
    result = run_training(config)
    
    if result["status"] == "success":
        print("Fine-tuning completed successfully!")
        return {
            "status": "success",
            "message": f"LoRA training completed for '{trigger_word}'",
            "models": result["models"],
            "image_count": image_count,
            "trigger_word": trigger_word,
            "steps": steps
        }
    else:
        return result

def setup_environment():
    """Setup training environment and dependencies"""
    print("Setting up training environment...")
    
    # Clone ai-toolkit if needed
    toolkit_path = "/tmp/ai-toolkit"
    if not os.path.exists(toolkit_path):
        print("Downloading ai-toolkit...")
        subprocess.run([
            "git", "clone", "https://github.com/ostris/ai-toolkit.git", toolkit_path
        ], check=True)
        subprocess.run([
            "git", "submodule", "update", "--init", "--recursive"
        ], cwd=toolkit_path, check=True)
    
    # Configure environment variables
    os.environ.update({
        'DISABLE_TELEMETRY': 'YES',
        'HF_TOKEN': os.getenv("HF_TOKEN"),
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True,max_split_size_mb:512',
        'TORCH_CUDNN_V8_API_ENABLED': '1',
        'NVIDIA_TF32_OVERRIDE': '1',
        'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': '1',
        'TOKENIZERS_PARALLELISM': 'false',
        'HF_HUB_ENABLE_HF_TRANSFER': '1'
    })
    
    import sys
    sys.path.insert(0, toolkit_path)

def process_dataset(image_zip, trigger_word, resolution):
    """Process uploaded dataset for training"""
    print("Processing training dataset...")
    
    # Setup directories
    base_dir = VOLUME_PATH
    dataset_dir = os.path.join(base_dir, "dataset")
    
    # Clean dataset directory
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)
    
    if not image_zip:
        print("No dataset provided, creating dummy data")
        return create_dummy_dataset(dataset_dir, trigger_word, resolution)
    
    # Download and extract dataset
    try:
        print(f"Downloading dataset: {image_zip}")
        zip_response = requests.get(image_zip, timeout=60)
        zip_response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
            temp_zip.write(zip_response.content)
            temp_zip_path = temp_zip.name
        
        # Extract and process images
        image_count = 0
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            with tempfile.TemporaryDirectory() as temp_extract_dir:
                zip_ref.extractall(temp_extract_dir)
                
                for root, dirs, files in os.walk(temp_extract_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                            if process_image(root, file, dataset_dir, trigger_word, resolution, image_count):
                                image_count += 1
        
        os.remove(temp_zip_path)
        print(f"Processed {image_count} training images")
        return image_count
        
    except Exception as e:
        print(f"Dataset processing failed: {e}")
        return 0

def process_image(root, filename, dataset_dir, trigger_word, resolution, image_count):
    """Process individual training image"""
    try:
        old_path = os.path.join(root, filename)
        with PILImage.open(old_path) as img:
            img = img.convert('RGB')
            img = img.resize((resolution, resolution), PILImage.Resampling.LANCZOS)
            
            new_filename = f"training_image_{image_count + 1}.jpg"
            new_path = os.path.join(dataset_dir, new_filename)
            img.save(new_path, 'JPEG', quality=95)
            
            # Create caption file
            caption_path = os.path.join(dataset_dir, f"training_image_{image_count + 1}.txt")
            with open(caption_path, 'w') as f:
                f.write(f"a photo of {trigger_word}")
            
            return True
    except Exception as e:
        print(f"Failed to process {filename}: {e}")
        return False

def create_dummy_dataset(dataset_dir, trigger_word, resolution):
    """Create dummy dataset for testing"""
    dummy_img = PILImage.new('RGB', (resolution, resolution), color='red')
    dummy_img.save(os.path.join(dataset_dir, "dummy.jpg"))
    
    with open(os.path.join(dataset_dir, "dummy.txt"), 'w') as f:
        f.write(f"a photo of {trigger_word}")
    
    return 1

def create_training_config(trigger_word, steps, learning_rate, rank, alpha, resolution):
    """Create training configuration"""
    base_dir = VOLUME_PATH
    dataset_dir = os.path.join(base_dir, "dataset")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        "job": "extension",
        "config": {
            "name": f"flux_lora_{trigger_word}_{resolution}",
            "process": [{
                "type": "sd_trainer",
                "training_folder": output_dir,
                "device": "cuda:0",
                "trigger_word": trigger_word,
                "network": {
                    "type": "lora",
                    "linear": rank,
                    "linear_alpha": alpha
                },
                "save": {
                    "dtype": "float16",
                    "save_every": steps // 2,
                    "max_step_saves_to_keep": 2
                },
                "datasets": [{
                    "folder_path": os.path.abspath(dataset_dir),
                    "caption_ext": "txt",
                    "caption_dropout_rate": 0.05,
                    "cache_latents": True,
                    "skip_cache_check": True,
                    "shuffle_tokens": False,
                    "cache_latents_to_disk": True,
                    "resolution": [resolution]
                }],
                "train": {
                    "batch_size": 1,
                    "steps": steps,
                    "gradient_accumulation_steps": 8,
                    "train_unet": True,
                    "train_text_encoder": False,
                    "gradient_checkpointing": False,
                    "noise_scheduler": "flowmatch",
                    "optimizer": "adamw8bit",
                    "lr": learning_rate,
                    "ema_config": {
                        "use_ema": True,
                        "ema_decay": 0.99
                    },
                    "dtype": "fp16"
                },
                "model": {
                    "name_or_path": "black-forest-labs/FLUX.1-dev",
                    "is_flux": True,
                    "quantize": True
                },
                "sample": {
                    "sampler": "flowmatch",
                    "sample_every": steps // 2,
                    "width": resolution,
                    "height": resolution,
                    "prompts": [
                        f"a photo of {trigger_word}",
                        f"{trigger_word} in professional lighting",
                        f"portrait of {trigger_word}, high quality"
                    ],
                    "neg": "",
                    "seed": 42,
                    "walk_seed": False,
                    "guidance_scale": 4,
                    "sample_steps": 10
                }
            }]
        }
    }
    
    return config

def run_training(config):
    """Execute the training process"""
    print("Starting LoRA training...")
    
    try:
        # Save config
        toolkit_path = "/tmp/ai-toolkit"
        config_path = os.path.join(toolkit_path, "config", "train_config.yaml")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Run training
        os.chdir(toolkit_path)
        result = subprocess.run(
            ["python", "run.py", "config/train_config.yaml"],
            capture_output=False,
            text=True,
            check=True
        )
        
        # Find trained models
        output_dir = os.path.join(VOLUME_PATH, "output")
        trained_models = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.safetensors'):
                    trained_models.append(os.path.join(root, file))
        
        return {
            "status": "success",
            "models": trained_models
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Training failed: {str(e)}"
        }