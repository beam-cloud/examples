from beam import Client
import os
import zipfile
import base64
from PIL import Image
from io import BytesIO
import time
import requests
import argparse

# Your Beam configuration
BEAM_TOKEN = "your_beam_token_here"
TRAIN_FUNCTION = "https://your-train-endpoint.app.beam.cloud"
GENERATE_FUNCTION = "https://your-generate-endpoint.app.beam.cloud"

def train(folder_path: str, trigger_word: str, steps: int = 1000, lr: float = 1e-4, rank: int = 16):
    """Train a LoRA using Beam SDK"""
    
    print(f"Training '{trigger_word}'")
    print(f"Using images from: {folder_path}")
    
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")
    
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {folder_path}")
    
    print(f"Found {len(image_files)} images")
    
    zip_path = f"{trigger_word}_training.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in image_files:
            file_path = os.path.join(folder_path, file)
            zipf.write(file_path, file)
            print(f"   Added: {file}")
    
    print("Uploading with Beam...")
    client = Client(token=BEAM_TOKEN)
    
    try:
        zip_url = client.upload_file(zip_path)
        print(f"Uploaded: {zip_url}")
        
        print("Starting training...")
        response = requests.post(TRAIN_FUNCTION, 
            headers={'Authorization': f'Bearer {BEAM_TOKEN}', 'Content-Type': 'application/json'},
            json={
                'image_zip': zip_url,
                'trigger_word': trigger_word,
                'steps': steps,
                'learning_rate': lr,
                'rank': rank
            })
        
        os.remove(zip_path)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Training started!")
            return {"status": "success", "result": result, "trigger_word": trigger_word}
        else:
            print(f"Failed: {response.status_code} - {response.text}")
            return {"status": "error", "message": response.text}
        
    except Exception as e:
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise Exception(f"Training failed: {e}")

def generate(prompt: str, lora_name: str = None, width: int = 1024, height: int = 1024, steps: int = 20, guidance: float = 4.0, seed: int = None, lora_scale: float = 0.8):
    """Generate image using requests like training"""
    
    print(f"Generating: '{prompt}'")
    
    params = {
        'prompt': prompt,
        'lora_name': lora_name,
        'width': width,
        'height': height,
        'steps': steps,
        'guidance': guidance,
        'seed': seed,
        'lora_scale': lora_scale
    }
    params = {k: v for k, v in params.items() if v is not None}
    
    try:
        print(f"Sending request (lora: {lora_name or 'None'})...")
        response = requests.post(
            GENERATE_FUNCTION,
            headers={'Authorization': f'Bearer {BEAM_TOKEN}', 'Content-Type': 'application/json'},
            json=params,
            timeout=600
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if 'task_id' in result and 'image' not in result:
                print(f"Task started: {result['task_id']}")
                print("Waiting for task to complete...")
                print("Image will be saved to Beam volume: ./flux-lora-data/")
                print("Check your Beam dashboard for progress")
                print("Task submitted successfully!")
                print("Check 'beam ls volume-name' to see generated images")
                return result
                        
            if result.get('image'):
                print("Saving image locally...")
                img_data = base64.b64decode(result['image'])
                img = Image.open(BytesIO(img_data))
                
                filename = f"generated_{lora_name or 'base'}_{int(time.time())}.png"
                full_path = os.path.abspath(filename)
                img.save(full_path)
                
                if os.path.exists(full_path):
                    print(f"Successfully saved locally: {full_path}")
                    
                    if result.get('url'):
                        print(f"Public URL: {result['url']}")
                    if result.get('settings'):
                        settings = result['settings']
                        print(f"Settings: {settings['width']}x{settings['height']}, steps: {settings['steps']}")
                else:
                    print(f"Failed to save file: {full_path}")
            else:
                print("No image data in response!")
                print(f"Full response: {result}")
            
            return result
        else:
            print(f"Failed: {response.status_code} - {response.text}")
            return {"status": "error", "message": response.text}
            
    except requests.exceptions.Timeout:
        print("Request timed out after 10 minutes")
        print("The generation might still be running on Beam")
        print("Check your Beam dashboard for task status")
        return {"status": "error", "message": "Request timed out"}
    except Exception as e:
        print(f"Generation failed: {e}")
        return {"status": "error", "message": str(e)}

def wait_for_task_completion(task_id: str, max_wait: int = 600, check_interval: int = 10):
    """Wait for a task to complete using Beam SDK"""
    
    print(f"Polling task {task_id} every {check_interval} seconds...")
    start_time = time.time()
    
    client = Client(token=BEAM_TOKEN)
    
    while time.time() - start_time < max_wait:
        try:
            task_result = client.get_task_result(task_id)
            
            if task_result is not None:
                print("Task completed successfully!")
                return task_result
            else:
                print("Task still running...")
                
        except Exception as e:
            print(f"Task running (checking via dashboard)... {int(time.time() - start_time)}s elapsed")
        
        time.sleep(check_interval)
    
    print(f"Timeout waiting for task completion after {max_wait} seconds")
    print("Check your Beam dashboard - the task may still be running")
    return {"status": "timeout", "task_id": task_id}

def wait_for_training(task_id: str, check_interval: int = 60):
    """Check training status by polling task_id"""
    
    print(f"Training task: {task_id}")
    print("Training typically takes 30-60 minutes depending on steps")
    print("Check your Beam dashboard for real-time progress:")
    print("https://dashboard.beam.cloud/")
    print("Once training completes, the models will be saved to your volume")
    
    return {"status": "info", "message": "Check Beam dashboard for progress", "task_id": task_id}

def download_image(filename: str = None):
    """Download the latest generated image from Beam volume"""
    
    if filename:
        try:
            print(f"Downloading {filename}...")
            local_path = f"./downloaded_{filename}"
            
            import subprocess
            result = subprocess.run([
                'beam', 'cp', f'beam://flux-lora-data/{filename}', local_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Downloaded: {local_path}")
                return {"status": "success", "path": local_path}
            else:
                print(f"Download failed: {result.stderr}")
                return {"status": "error", "message": result.stderr}
                
        except Exception as e:
            print(f"Download failed: {e}")
            return {"status": "error", "message": str(e)}
    else:
        print("Finding latest generated image...")
        print("Use: beam ls volume-name")
        print("Then: python client.py download --filename generated_xxx.png")
        return {"status": "info", "message": "Specify filename to download"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLUX LoRA Training & Generation (Beam SDK)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Train Command ---
    train_parser = subparsers.add_parser("train", help="Train a LoRA model.")
    train_parser.add_argument("folder", help="Path to the folder with training images.")
    train_parser.add_argument("trigger_word", help="The trigger word for the LoRA (e.g., 'iraTok').")
    train_parser.add_argument("--steps", type=int, default=1650, help="Number of training steps (formula: n*100+350, where n=images).")
    train_parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate.")
    train_parser.add_argument("--rank", type=int, default=32, help="LoRA rank.")

    # --- Generate Command ---
    gen_parser = subparsers.add_parser("generate", help="Generate an image.")
    gen_parser.add_argument("prompt", help="The text prompt for generation.")
    gen_parser.add_argument("--lora", dest="lora_name", help="Name of the LoRA to use (your trigger_word).")
    gen_parser.add_argument("--width", type=int, default=1024)
    gen_parser.add_argument("--height", type=int, default=1024)
    gen_parser.add_argument("--steps", type=int, default=35, help="Number of inference steps.")
    gen_parser.add_argument("--guidance", type=float, default=3)
    gen_parser.add_argument("--seed", type=int)
    gen_parser.add_argument("--lora-scale", type=float, default=0.9, help="LoRA adapter strength (0.0-1.0)")
    

    # --- Wait Command ---
    wait_parser = subparsers.add_parser("wait", help="Check the status of a training task.")
    wait_parser.add_argument("task_id", help="The task ID to check.")
    
    # --- Download Command ---
    download_parser = subparsers.add_parser("download", help="Download the latest generated image from volume.")
    download_parser.add_argument("--filename", help="Specific filename to download (optional)")

    args = parser.parse_args()
    
    if args.command == "train":
        result = train(
            folder_path=args.folder,
            trigger_word=args.trigger_word,
            steps=args.steps,
            lr=args.lr,
            rank=args.rank
        )
        print(f"\nResult: {result}")
        
    elif args.command == "generate":
        result = generate(
            prompt=args.prompt,
            lora_name=args.lora_name,
            width=args.width,
            height=args.height,
            steps=args.steps,
            guidance=args.guidance,
            seed=args.seed,
            lora_scale=args.lora_scale
        )
        print(f"\nResult: {result}")
        
    elif args.command == "wait":
        result = wait_for_training(args.task_id)
        print(f"\nResult: {result}")
        
    elif args.command == "download":
        result = download_image(args.filename)
        print(f"\nResult: {result}")
        
    else:
        print("Invalid command. Use --help to see available commands.")
        sys.exit(1)
