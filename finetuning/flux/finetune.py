from beam import function, Volume, Image
import os
import requests
import zipfile
import tempfile
import yaml
from PIL import Image as PILImage

VOLUME_PATH = "./flux-lora-data"

@function(
    name="simple-flux-train",
    gpu="H100",
    cpu=4,
    memory="32Gi",
    timeout=3600,
    volumes=[Volume(name="flux-lora-data", mount_path=VOLUME_PATH)],
    image=Image(python_version="python3.12")
        .add_commands([
            "apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1",
            "git clone https://github.com/ostris/ai-toolkit.git /ai-toolkit",
            "cd /ai-toolkit && git submodule update --init --recursive",
            "pip3.12 install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126",
            "pip3.12 install pyyaml requests pillow opencv-python-headless",
            "cd /ai-toolkit && pip3.12 install -r requirements.txt"
        ]),
    secrets=["HF_TOKEN"]
)

def train_flux_lora(
    image_zip: str,
    trigger_word: str = "TOK",
    steps: int = 1500,
    learning_rate: float = 4e-4,
    rank: int = 32
):
    """
    Train FLUX LoRA
    """
    print(f"Training '{trigger_word}' LoRA")
    
    dataset_dir, image_count = setup_dataset(image_zip, trigger_word)
    
    optimal_steps = (image_count * 100) + 350 
    print(f"Found {image_count} images")
    print(f"Adjusted steps: {optimal_steps}")
    steps = optimal_steps
    
    config = {
        "job": "extension",
        "config": {
            "name": f"flux_lora_{trigger_word}",
            "process": [{
                "type": "sd_trainer",
                "training_folder": VOLUME_PATH,
                "device": "cuda:0",
                "trigger_word": trigger_word,
                "network": {
                    "type": "lora",
                    "linear": 32, 
                    "linear_alpha": 32,
                    "network_kwargs": {
                        "only_if_contains": [
                            "transformer.single_transformer_blocks.7",  
                            "transformer.single_transformer_blocks.12",  
                            "transformer.single_transformer_blocks.16",
                            "transformer.single_transformer_blocks.20"   
                        ]
                    }
                },
                "save": {
                    "dtype": "float16",
                    "save_every": 10000,
                    "max_step_saves_to_keep": 4,
                    "push_to_hub": False
                },
                "datasets": [{
                    "folder_path": "/ai-toolkit/input",
                    "caption_ext": "txt",
                    "caption_dropout_rate": 0.05,
                    "shuffle_tokens": False,
                    "cache_latents_to_disk": True,
                    "resolution": [768, 1024]
                }],
                "train": {
                    "batch_size": 1,
                    "steps": steps,
                    "gradient_accumulation_steps": 1,
                    "train_unet": True,
                    "train_text_encoder": False,
                    "gradient_checkpointing": False,
                    "noise_scheduler": "flowmatch",
                    "optimizer": "adamw8bit",
                    "lr": 4e-4,
                    "lr_scheduler": "cosine",
                    "skip_first_sample": True,
                    "disable_sampling": True,
                    "ema_config": {
                        "use_ema": True,
                        "ema_decay": 0.99
                    },
                    "dtype": "bf16"
                },
                "model": {
                    "name_or_path": "black-forest-labs/FLUX.1-dev",
                    "is_flux": True,
                    "quantize": False,
                    "low_vram": False
                },
                "sample": {
                    "sampler": "flowmatch",
                    "sample_every": 10000,
                    "width": 1024,
                    "height": 1024,
                    "prompts": [f"portrait of {trigger_word} woman"],
                    "neg": "",
                    "seed": 42,
                    "walk_seed": True,
                    "guidance_scale": 3.5,
                    "sample_steps": 28
                }
            }]
        },
        "meta": {
            "name": f"flux_lora_{trigger_word}",
            "version": "1.0"
        }
    }
    
    config_path = "/ai-toolkit/train_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Config saved to: {config_path}")
    
    import subprocess
    
    env = os.environ.copy()
    env.update({
        'HF_TOKEN': os.getenv('HF_TOKEN'),
        'CUDA_VISIBLE_DEVICES': '0'
    })
    
    print("Starting training...")
    process = subprocess.Popen([
        "python3.12", "/ai-toolkit/run.py", "/ai-toolkit/train_config.yaml"
    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    
    for line in process.stdout:
        print(line.rstrip())
    
    return_code = process.wait()
    
    print("Ensuring models are saved to persistent volume...")
    copy_models_to_volume()
    
    if return_code == 0:
        return {
            "status": "success",
            "message": f"Training completed for {trigger_word}",
            "output": "Training completed successfully"
        }
    else:
        return {
            "status": "error", 
            "message": f"Training failed with return code {return_code}"
        }

def copy_models_to_volume():
    """Copy any models from ai-toolkit output to persistent volume"""
    import shutil
    
    source_dir = "/ai-toolkit/output"
    dest_dir = VOLUME_PATH
    
    if os.path.exists(source_dir):
        print(f"Copying models from {source_dir} to {dest_dir}")
        
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.safetensors') or file.endswith('.yaml') or file.endswith('.json'):
                    source_file = os.path.join(root, file)
                    # Create relative path structure in destination
                    rel_path = os.path.relpath(root, source_dir)
                    dest_folder = os.path.join(dest_dir, rel_path) if rel_path != '.' else dest_dir
                    os.makedirs(dest_folder, exist_ok=True)
                    
                    dest_file = os.path.join(dest_folder, file)
                    shutil.copy2(source_file, dest_file)
                    print(f"Copied: {file}")
    else:
        print(f"No source directory {source_dir} found")

def setup_dataset(image_zip, trigger_word):
    """Download and setup dataset in the format they expect"""
    dataset_dir = "/ai-toolkit/input"
    os.makedirs(dataset_dir, exist_ok=True)
    
    response = requests.get(image_zip)
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
        f.write(response.content)
        zip_path = f.name
    
    # Extract images and create captions    
    count = 0
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                zip_ref.extract(file, dataset_dir)
                
                base_name = os.path.splitext(file)[0]
                caption_path = os.path.join(dataset_dir, f"{base_name}.txt")
                
                # Only generate caption if .txt file doesn't exist
                if not os.path.exists(caption_path):
                    captions = [
                        f"portrait of {trigger_word} woman with long brown hair, looking at camera",
                        f"photo of {trigger_word} woman, long brown hair, natural lighting",
                        f"{trigger_word} woman with long brown hair, outdoor setting",
                        f"close-up portrait of {trigger_word}, long brown hair, detailed face",
                        f"{trigger_word} woman sitting, long brown hair, realistic photo",
                        f"portrait photo of {trigger_word} with long brown hair",
                        f"{trigger_word} woman, long brown hair, professional lighting",
                        f"photo of {trigger_word} woman, detailed facial features",
                        f"{trigger_word} with long brown hair, natural expression",
                        f"portrait of {trigger_word} woman, high quality photo"
                    ]
                    
                    caption = captions[count % len(captions)]
                    
                    with open(caption_path, 'w') as caption_file:
                        caption_file.write(caption)
                count += 1
    
    os.unlink(zip_path)
    print(f"Setup {count} training images in {dataset_dir}")
    return dataset_dir, count
