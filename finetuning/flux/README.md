# FLUX LoRA Fine-tuning on Beam

Implementation for fine-tuning FLUX models with LoRA using your own image datasets.

## Quick Start

1. **Create volume** (one-time setup):
   ```bash
   beam volume create flux-lora-finetune
   ```

2. **Deploy endpoints**:
   ```bash
   beam deploy finetune.py:train_lora --name train-lora
   beam deploy inference.py:generate --name generate-image
   ```

3. **Update configuration** in `upload.py`:
   - Set your `BEAM_TOKEN`
   - Update `ENDPOINTS` with your deployed URLs

4. **Train a model**:
   ```bash
   python upload.py train ./your_images my_concept
   ```

5. **Generate images**:
   ```bash
   python upload.py generate "a photo of my_concept" my_concept
   ```

## Volume Storage

The system uses a persistent Beam volume named `flux-lora-finetune` that automatically stores:

```
flux-lora-finetune/
├── dataset/        # Training datasets
└── output/         # Trained models
```

## File Structure

```
flux/
├── finetune.py     # Training endpoint
├── inference.py    # Image generation endpoint  
├── upload.py       # Local dataset handling
└── README.md       # This file
```

## Training

### Deploy Training Endpoint

```bash
beam deploy finetune.py:train_lora --name train-lora
```

### Start Training with Local Images

```python
from upload import upload_dataset, start_training

# Upload your dataset
result = upload_dataset("./training_images", "my_concept")

# Start training
training = start_training(
    result["zip_url"], 
    "my_concept",
    steps=1000,
    resolution=1024
)
```

### Training Parameters

- `trigger_word`: Token to associate with your concept (e.g., "my_dog", "abstract_art")
- `steps`: Training steps (default: 1000)
- `learning_rate`: Learning rate (default: 4e-4)
- `rank`: LoRA rank - higher = more capacity (default: 32)
- `alpha`: LoRA alpha scaling (default: 32)
- `resolution`: Image resolution (default: 1024)

## Image Generation

### Deploy Inference Endpoint

```bash
beam deploy inference.py:generate --name generate-image
```

### Generate Images

```python
from upload import generate_image

result = generate_image(
    "a sample of my_concept in sunlight",
    "my_concept",
    width=1024,
    height=1024,
    steps=20,
    seed=42
)
```

### Generation Parameters

- `prompt`: Text description of desired image
- `trigger_word`: Token used during training
- `width/height`: Image dimensions (256-1024)
- `num_inference_steps`: Denoising steps (1-50)
- `guidance_scale`: Prompt following strength (1.0-20.0)
- `seed`: Random seed for reproducibility
- `negative_prompt`: What to avoid
- `num_images`: Number to generate (1-4)

## Complete Workflow

```python
from upload import full_workflow

# Upload, train, and optionally test
result = full_workflow(
    local_folder="./my_images",
    trigger_word="my_style",
    test_prompt="a painting in my_style",
    steps=1000,
    resolution=1024
)
```

## Command Line Usage

```bash
# Upload dataset only
python upload.py upload ./images my_concept

# Upload and start training  
python upload.py train ./images my_concept

# Generate image (after training completes)
python upload.py generate "prompt text" my_concept

# Complete workflow
python upload.py workflow ./images my_concept "test prompt"
```

## Dataset Preparation

**Supported formats:** JPG, JPEG, PNG, WebP, BMP

**Recommendations:**
- 10-50 high-quality images work well
- Images will be resized to training resolution
- Varied poses/angles improve results
- Consistent lighting/style helps

**Example folder structure:**
```
training_images/
├── image1.jpg
├── image2.png
├── image3.jpg
└── ...
```

## Configuration

**Before using, update `upload.py`:**

```python
# Your Beam authentication token
BEAM_TOKEN = "<your_beam_token_here>"

# Your deployed endpoint URLs
ENDPOINTS = {
    "train": "https://train-lora-<your-endpoint>.app.beam.cloud",
    "inference": "https://generate-image-<your-endpoint>.app.beam.cloud"
}
```

## Monitoring

- **Beam Dashboard**: https://app.beam.cloud/
- **Training logs**: Available in Beam dashboard
- **Model files**: Stored in persistent volume `flux-lora-finetune`

## Tips

**For better results:**
```
- Use descriptive trigger words (e.g., "vintage_car" vs "car")
- Include trigger word in generation prompts
- Experiment with guidance_scale (3-15 range)
- Try different seeds for variety
```

**Training tips:**
```
- More steps = better quality but longer training
- Higher rank = more capacity but larger files
- Start with 512px for testing, use 768px or 1024px for final training (better quality)
```

**Generation tips:**
```
- Start with 20 inference steps
- Use guidance_scale 7-10 for most prompts
- Add negative prompts to avoid unwanted elements
```