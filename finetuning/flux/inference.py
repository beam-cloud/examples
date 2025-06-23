from beam import endpoint, Volume, Image, QueueDepthAutoscaler, Output
import torch
from diffusers import FluxPipeline
import os
from io import BytesIO
import base64

VOLUME_PATH = "./flux-lora-finetune"

# Global pipeline variable
pipeline = None

def load_pipeline():
    """Load FLUX pipeline with trained LoRA"""
    global pipeline
    
    if pipeline is not None:
        return pipeline
    
    print("Loading FLUX pipeline...")
    
    try:
        # Load base FLUX model with memory optimizations
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16,
            token=os.getenv("HF_TOKEN"),
            use_safetensors=True
        ).to("cuda")
        
        # Load trained LoRA if available
        output_dir = os.path.join(VOLUME_PATH, "output")
        print(f"Looking for LoRA models in: {output_dir}")
        print(f"Volume path: {VOLUME_PATH}")
        print(f"Volume exists: {os.path.exists(VOLUME_PATH)}")
        
        if os.path.exists(output_dir):
            print(f"Output directory exists: {output_dir}")
            print(f"Output directory contents:")
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path):
                    print(f"  DIR: {item}/")
                else:
                    print(f"  FILE: {item}")
            
            lora_files = find_lora_models(output_dir)
            print(f"Found {len(lora_files)} LoRA files")
            
            if lora_files:
                # Print all found files for debugging
                for i, f in enumerate(lora_files):
                    print(f"  {i+1}. {os.path.basename(f)}")
                
                final_models = [f for f in lora_files if not any(x in os.path.basename(f) for x in ['_0', 'checkpoint'])]
                if final_models:
                    latest_lora = max(final_models, key=os.path.getctime)
                    print(f"Selected final model: {os.path.basename(latest_lora)}")
                    print(f"Model creation time: {os.path.getctime(latest_lora)}")
                else:
                    latest_lora = max(lora_files, key=os.path.getctime)
                    print(f"Selected checkpoint model: {os.path.basename(latest_lora)}")
                    print(f"Model creation time: {os.path.getctime(latest_lora)}")
                
                load_lora_weights(pipeline, latest_lora)
                print(f"Model loaded from: {os.path.relpath(latest_lora, VOLUME_PATH)}")
            else:
                print("No LoRA models found - using base FLUX model")
                if os.path.exists(output_dir):
                    print("Contents of output directory:")
                    for root, dirs, files in os.walk(output_dir):
                        for file in files:
                            print(f"  - {os.path.join(root, file)}")
        else:
            print(f"Output directory does not exist: {output_dir}")
            print(f"Volume contents:")
            if os.path.exists(VOLUME_PATH):
                for root, dirs, files in os.walk(VOLUME_PATH):
                    for file in files:
                        print(f"  - {os.path.join(root, file)}")
        
        print("Pipeline loaded successfully!")
        return pipeline
        
    except Exception as e:
        print(f"Failed to load pipeline: {e}")
        raise

def find_lora_models(output_dir):
    """Find available LoRA model files"""
    lora_files = []
    print(f"Searching for LoRA files in: {output_dir}")
    
    if not os.path.exists(output_dir):
        print(f"Output directory does not exist: {output_dir}")
        return lora_files
    
    for root, dirs, files in os.walk(output_dir):
        print(f"Checking directory: {root}")
        for file in files:
            print(f"   Found file: {file}")
            if file.endswith('.safetensors'):
                if not '_000000' in file and not file.endswith('_0.safetensors'):
                    lora_files.append(os.path.join(root, file))
                    print(f"     Added LoRA file: {file}")
                else:
                    print(f"     Skipped checkpoint: {file}")
            elif file.endswith('.bin') or file.endswith('.pt'):
                # Also check for other common LoRA formats
                lora_files.append(os.path.join(root, file))
                print(f"     Added LoRA file: {file}")
    
    print(f"Total LoRA files found: {len(lora_files)}")
    return lora_files

def load_lora_weights(pipeline, lora_path):
    """Load LoRA weights into pipeline"""
    try:
        pipeline.load_lora_weights(
            os.path.dirname(lora_path),
            weight_name=os.path.basename(lora_path)
        )
        print(f"Loaded LoRA: {os.path.basename(lora_path)}")
    except Exception as e:
        print(f"Failed to load LoRA: {e}")
        print(f"   Attempted to load from: {lora_path}")

@endpoint(
    name="generate-image",
    on_start=load_pipeline,
    gpu="A100-40",
    cpu=4,
    memory="32Gi",
    image=Image(python_version="python3.11")
        .add_python_packages([
            "torch==2.6.0",
            "diffusers",
            "transformers==4.52.4",
            "safetensors",
            "accelerate",
            "pillow",
            "hf_transfer",
            "protobuf",
            "sentencepiece"
        ])
        .with_envs("HF_HUB_ENABLE_HF_TRANSFER=1"),
    volumes=[Volume(name="flux-lora-finetune", mount_path=VOLUME_PATH)],
    secrets=["HF_TOKEN"],
    autoscaler=QueueDepthAutoscaler(max_containers=1, tasks_per_container=1),
    keep_warm_seconds=60
)
def generate(
    prompt: str,
    trigger_word: str = "TOK",
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    seed: int = None,
    negative_prompt: str = "",
    num_images: int = 1
):
    """
    Generate images using fine-tuned FLUX LoRA model
    
    Args:
        prompt: Text description of desired image
        trigger_word: Token used during training
        width: Image width (256-1024)
        height: Image height (256-1024) 
        num_inference_steps: Number of denoising steps (1-50)
        guidance_scale: How closely to follow prompt (1.0-20.0)
        seed: Random seed for reproducibility
        negative_prompt: What to avoid in generation
        num_images: Number of images to generate (1-4)
    """
    global pipeline
    
    try:
        # Ensure pipeline is loaded
        if pipeline is None:
            pipeline = load_pipeline()
        
        # Validate parameters
        num_images = max(1, min(num_images, 4))
        width = max(256, min(width, 1024))
        height = max(256, min(height, 1024))
        num_inference_steps = max(1, min(num_inference_steps, 50))
        guidance_scale = max(1.0, min(guidance_scale, 20.0))
        
        print(f"Generating {num_images} image(s)")
        print(f"Prompt: '{prompt}'")
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Generate images
        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
                generator=torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None
            )
        
        # Convert images to base64 and create shareable URLs
        encoded_images = []
        image_urls = []
        
        for i, image in enumerate(result.images):
            # Base64 encoding for immediate use
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            encoded_images.append(img_str)
            
            # Also create shareable URL using Beam Output
            try:
                filename = f"generated_image_{i+1}.png"
                output = Output(path=filename)
                output.save(image)
                image_urls.append(output.public_url)
            except Exception as e:
                print(f"Failed to create public URL for image {i+1}: {e}")
                image_urls.append(None)
        
        print(f"Generated {len(encoded_images)} image(s)")
        
        return {
            "status": "success",
            "images": encoded_images,  # Base64 for immediate use
            "image_urls": image_urls,  # Public URLs for sharing
            "prompt": prompt,
            "trigger_word": trigger_word,
            "seed": seed,
            "settings": {
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "negative_prompt": negative_prompt
            },
            "num_images": len(encoded_images)
        }
        
    except Exception as e:
        print(f"Generation failed: {e}")
        return {
            "status": "error",
            "message": f"Failed to generate image: {str(e)}",
            "prompt": prompt
        }