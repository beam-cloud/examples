from beam import Client
import os
import zipfile
import time
from datetime import datetime

# Your Beam authentication token
BEAM_TOKEN = "<your_beam_token_here>"

# Your deployed endpoint URLs
ENDPOINTS = {
    "train": "https://train-lora-<your-endpoint>.app.beam.cloud",
    "inference": "https://generate-image-<your-endpoint>.app.beam.cloud"
}

def upload_dataset(local_folder: str, trigger_word: str):
    """
    Upload local image dataset to Beam for training
    
    Args:
        local_folder: Path to folder containing training images
        trigger_word: Token to associate with your concept
    
    Returns:
        dict: Upload result with zip URL
    """
    print(f"Preparing dataset from: {local_folder}")
    print(f"Trigger word: '{trigger_word}'")
    
    # Validate inputs
    if not os.path.exists(local_folder):
        raise ValueError(f"❌ Folder not found: {local_folder}")
    
    if not trigger_word or trigger_word.strip() == "":
        raise ValueError("❌ Trigger word cannot be empty!")
    
    # Initialize Beam client
    client = Client(token=BEAM_TOKEN)
    
    # Create zip file from local images
    zip_filename = f"{trigger_word.replace(' ', '_')}_dataset.zip"
    image_count = create_dataset_zip(local_folder, zip_filename)
    
    if image_count == 0:
        raise ValueError(f"❌ No images found in {local_folder}")
    
    try:
        print("Uploading dataset to Beam...")
        zip_url = client.upload_file(zip_filename)
        print(f"Dataset uploaded successfully!")
        print(f"{image_count} images ready for training")
        
        # Clean up local zip
        os.remove(zip_filename)
        
        return {
            "status": "success",
            "zip_url": zip_url,
            "image_count": image_count,
            "trigger_word": trigger_word
        }
        
    except Exception as e:
        # Clean up on failure
        if os.path.exists(zip_filename):
            os.remove(zip_filename)
        raise Exception(f"❌ Upload failed: {e}")

def create_dataset_zip(local_folder: str, zip_filename: str):
    """Create zip file from local image folder"""
    image_count = 0
    supported_formats = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename in os.listdir(local_folder):
            if filename.lower().endswith(supported_formats):
                file_path = os.path.join(local_folder, filename)
                if os.path.isfile(file_path):
                    zipf.write(file_path, filename)
                    image_count += 1
                    print(f"  Added: {filename}")
    
    print(f"Created dataset zip with {image_count} images")
    return image_count

def start_training(zip_url: str, trigger_word: str, **kwargs):
    """
    Start LoRA training with uploaded dataset
    
    Args:
        zip_url: URL to uploaded dataset zip
        trigger_word: Token used during training
        **kwargs: Additional training parameters
    
    Returns:
        dict: Training request result
    """
    import requests
    
    # Default training parameters
    training_params = {
        "image_zip": zip_url,
        "trigger_word": trigger_word,
        "steps": kwargs.get("steps", 1000),
        "learning_rate": kwargs.get("learning_rate", 4e-4),
        "rank": kwargs.get("rank", 32),
        "alpha": kwargs.get("alpha", 32),
        "resolution": kwargs.get("resolution", 1024)
    }
    
    print("Starting LoRA training...")
    print(f"Steps: {training_params['steps']}")
    print(f"Resolution: {training_params['resolution']}x{training_params['resolution']}")
    
    try:
        response = requests.post(
            ENDPOINTS["train"],
            headers={
                'Authorization': f'Bearer {BEAM_TOKEN}',
                'Content-Type': 'application/json'
            },
            json=training_params,
            timeout=300
        )
        
        if response.status_code == 200:
            print("Training started successfully!")
            return {"status": "training_started", "params": training_params}
        else:
            print(f"HTTP {response.status_code} - Check Beam dashboard")
            return {"status": "request_sent", "params": training_params}
            
    except requests.exceptions.Timeout:
        print("Request timed out - Training likely started")
        return {"status": "timeout", "params": training_params}
    except Exception as e:
        print(f"Request error: {e}")
        return {"status": "error", "message": str(e)}

def generate_image(prompt: str, trigger_word: str, **kwargs):
    """
    Generate image using trained model
    
    Args:
        prompt: Text description of desired image
        trigger_word: Token used during training
        **kwargs: Additional generation parameters
    
    Returns:
        dict: Generation result with base64 images
    """
    import requests
    import base64
    from PIL import Image
    from io import BytesIO
    
    # Default generation parameters
    gen_params = {
        "prompt": prompt,
        "trigger_word": trigger_word,
        "width": kwargs.get("width", 512),
        "height": kwargs.get("height", 512),
        "num_inference_steps": kwargs.get("steps", 20),
        "guidance_scale": kwargs.get("guidance_scale", 7.5),
        "seed": kwargs.get("seed"),
        "negative_prompt": kwargs.get("negative_prompt", ""),
        "num_images": kwargs.get("num_images", 1)
    }
    
    print(f"Generating image...")
    print(f"Prompt: '{prompt}'")
    
    try:
        response = requests.post(
            ENDPOINTS["inference"],
            headers={
                'Authorization': f'Bearer {BEAM_TOKEN}',
                'Content-Type': 'application/json'
            },
            json=gen_params,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                print("Image generated successfully!")
                
                # Save first image locally
                if result.get("images"):
                    image_data = base64.b64decode(result["images"][0])
                    image = Image.open(BytesIO(image_data))
                    
                    filename = f"generated_{int(time.time())}.png"
                    image.save(filename)
                    print(f"Saved: {filename}")
                
                return result
            else:
                print(f"Generation failed: {result.get('message', 'Unknown error')}")
                return result
        else:
            print(f"HTTP {response.status_code}: {response.text}")
            return {"status": "error", "message": f"HTTP {response.status_code}"}
            
    except Exception as e:
        print(f"Generation failed: {e}")
        return {"status": "error", "message": str(e)}

def full_workflow(local_folder: str, trigger_word: str, test_prompt: str = None, **training_kwargs):
    """
    Complete workflow: upload dataset, train model, and test generation
    
    Args:
        local_folder: Path to training images
        trigger_word: Token for your concept
        test_prompt: Optional test prompt after training
        **training_kwargs: Training parameters
    
    Returns:
        dict: Complete workflow results
    """
    workflow_start = datetime.now()
    
    print("Starting complete LoRA workflow...")
    print("="*50)
    
    try:
        # Step 1: Upload dataset
        print("STEP 1: Uploading dataset")
        upload_result = upload_dataset(local_folder, trigger_word)
        
        # Step 2: Start training
        print("\nSTEP 2: Starting training")
        training_result = start_training(
            upload_result["zip_url"], 
            trigger_word, 
            **training_kwargs
        )
        
        # Step 3: Optional test generation
        if test_prompt:
            print(f"\nSTEP 3: Testing with prompt: '{test_prompt}'")
            # Note: In real usage, you'd wait for training to complete first
            print("(Wait for training to complete before testing)")
        
        total_time = datetime.now() - workflow_start
        
        print("\nWorkflow completed!")
        print(f"Total time: {str(total_time).split('.')[0]}")
        
        return {
            "status": "success",
            "upload": upload_result,
            "training": training_result,
            "total_time": str(total_time).split('.')[0]
        }
        
    except Exception as e:
        print(f"\nWorkflow failed: {e}")
        return {"status": "error", "message": str(e)}

# Usage examples
if __name__ == "__main__":
    import sys
    
    # Update these before running
    print("Remember to update BEAM_TOKEN and ENDPOINTS in this file!")
    
    if len(sys.argv) < 2:
        print("Usage examples:")
        print("  python upload.py upload ./images my_concept")
        print("  python upload.py train ./images my_concept")
        print("  python upload.py generate 'a photo of my_concept' my_concept")
        print("  python upload.py workflow ./images my_concept 'test prompt'")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "upload" and len(sys.argv) >= 4:
        folder = sys.argv[2]
        trigger = sys.argv[3]
        result = upload_dataset(folder, trigger)
        print("Result:", result)
        
    elif command == "train" and len(sys.argv) >= 4:
        folder = sys.argv[2] 
        trigger = sys.argv[3]
        upload_result = upload_dataset(folder, trigger)
        training_result = start_training(upload_result["zip_url"], trigger)
        print("Training result:", training_result)
        
    elif command == "generate" and len(sys.argv) >= 4:
        prompt = sys.argv[2]
        trigger = sys.argv[3]
        result = generate_image(prompt, trigger)
        print("Generation result:", result)
        
    elif command == "workflow" and len(sys.argv) >= 4:
        folder = sys.argv[2]
        trigger = sys.argv[3]
        test_prompt = sys.argv[4] if len(sys.argv) > 4 else None
        result = full_workflow(folder, trigger, test_prompt)
        print("Workflow result:", result)
        
    else:
        print("Invalid command or missing arguments")