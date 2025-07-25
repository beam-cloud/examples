# FLUX LoRA Fine-tuning on Beam

Fast and efficient FLUX LoRA training with optimal parameters for small datasets (10-30 images).

## Quick Start

```bash
# 1. Create volume (one-time setup)
beam volume create flux-lora-data

# 2. Deploy training endpoint
beam deploy finetune.py:train_flux_lora

# 3. Deploy inference endpoint  
beam deploy inference.py:generate_image

# 4. Update your tokens in upload.py
# 5. Train your model
python upload.py train ./your_images yourTriggerWord

# 6. Generate images
python upload.py generate "photo of yourTriggerWord woman" --lora yourTriggerWord
```

## Volume Storage

The system uses persistent volume `flux-lora-data`:

```
flux-lora-data/
├── flux_lora_yourTriggerWord/     # Trained LoRA models
│   └── flux_lora_yourTriggerWord.safetensors
├── hf_cache/                      # Cached base models
└── generated_*.png                # Generated images
```

## File Structure

```
flux/
├── finetune.py     # Training function (@function)
├── inference.py    # Generation function (@function)
├── upload.py       # Client interface (like client.py)
└── README.md       # This guide
```

## Training

### Optimal Training Parameters

**Perfect for small datasets (10-30 images):**
```
Images: 10-30 high-quality photos
Steps: Auto-calculated (images × 100 + 350)
Training time: ~10 minutes for 13 images
VRAM: ~32GB (H100 GPU)
Resolution: 768px + 1024px mixed
```

### Deploy Training Endpoint

```bash
beam deploy finetune.py:train_flux_lora
```

### Start Training

```bash
python upload.py train ./your_images yourTriggerWord
```

**Example:**
```bash
python upload.py train ./ira_photos irunTok
```

### Advanced Training Options

```bash
python upload.py train ./photos myTrigger --steps 1650 --lr 4e-4 --rank 32
```

**Training Parameters:**
```
--steps: Training steps (default: auto-calculated)
--lr: Learning rate (default: 4e-4)
--rank: LoRA rank (default: 32)
```

## Dataset Preparation

### Image Requirements

```
Format: JPG, JPEG, PNG, WebP
Count: 10-30 images (LoRA works great with just 10-13 images, no need for more than 30)
Quality: High-resolution, varied poses
Consistency: Similar lighting/style
```

### Caption Handling

The system automatically creates captions for your images:

```python
# Auto-generated captions include:
"portrait of yourTriggerWord woman with long brown hair, looking at camera"
"photo of yourTriggerWord woman, natural lighting"
"yourTriggerWord woman with long brown hair, outdoor setting"
```

**Custom captions:** Create `.txt` files with same name as images:
```
your_folder/
├── photo1.jpg
├── photo1.txt          # "portrait of myTrigger woman smiling"
├── photo2.jpg
└── photo2.txt          # "myTrigger woman in professional attire"
```

## Image Generation

### Deploy Inference Endpoint

```bash
beam deploy inference.py:generate_image
```

### Generate Images

```bash
python upload.py generate "your prompt here" --lora yourTriggerWord
```

**Example:**
```bash
python upload.py generate "photo of irunTok woman with brown hair in Paris, on the background Eiffel Tower, high quality" --lora irunTok
```

### Generation Parameters

```bash
python upload.py generate "prompt" --lora triggerWord \
  --width 1024 --height 1024 --steps 35 --guidance 3.0 --seed 42 --lora-scale 0.9
```

**Parameters:**
```
--lora: Your trigger word (LoRA name)
--width/height: Image dimensions (default: 1024x1024)
--steps: Inference steps (default: 35)
--guidance: Prompt adherence (default: 3.0)
--seed: Random seed for reproducibility
--lora-scale: LoRA strength 0.0-1.0 (default: 0.9)
```

## Command Reference

### Training Commands

```bash
# Basic training
python upload.py train ./photos triggerWord

# Custom parameters
python upload.py train ./photos triggerWord --steps 1650 --lr 4e-4

# Check training status
python upload.py wait <task_id>
```

### Generation Commands

```bash
# Generate with LoRA
python upload.py generate "prompt" --lora triggerWord

# Generate without LoRA (base model)
python upload.py generate "prompt"

# Custom generation settings
python upload.py generate "prompt" --lora triggerWord --width 512 --height 768 --steps 20
```

### File Management

```bash
# List volume contents
beam ls flux-lora-data

# Download generated image
beam cp beam://flux-lora-data/generated_random_9384.png ./

# Download specific file
python upload.py download --filename generated_random_9384.png
```

## Configuration

**Update `upload.py` with your credentials:**

```python
# Replace with your actual tokens
BEAM_TOKEN = "your_beam_token_here"
TRAIN_FUNCTION = "https://your-train-endpoint.app.beam.cloud"
GENERATE_FUNCTION = "https://your-generate-endpoint.app.beam.cloud"
```

## Step Calculation Formula

**Automatic step calculation:**
```python
optimal_steps = (image_count × 100) + 350

# Examples:
# 5  images = 850 steps (~5 minutes)
# 10 images = 1,350 steps (~8 minutes)
# 13 images = 1,650 steps (~10 minutes)  
# 20 images = 2,350 steps (~15 minutes)
# 30 images = 3,350 steps (~20 minutes)
```

## Tips & Best Practices

### Training Tips

```
Use 10-30 images for best results
Varied poses and angles improve quality
Consistent lighting helps training
Training completes in ~10 minutes for 13 images
H100 GPU provides optimal performance
```

### Generation Tips

```
Always include your trigger word in prompts
Start with guidance_scale 3.0-4.0
Use 28-35 inference steps for quality
Experiment with lora_scale 0.7-1.0
Try different seeds for variety
```

### Prompt Examples

```bash
# Portrait style
"portrait of yourTrigger woman, professional lighting, high quality"

# Specific scenes
"photo of yourTrigger woman in Paris, Eiffel Tower background"

# Artistic styles
"yourTrigger woman in the style of Renaissance painting"

# Different settings
"yourTrigger woman at sunset, golden hour lighting"
```

## Troubleshooting

### Training Issues

```
Check HF_TOKEN is set correctly
Verify images are valid formats (JPG, PNG)
Ensure 10-30 images in folder
Monitor training via Beam dashboard
```

### Generation Issues

```
Verify LoRA name matches training trigger word
Check if training completed successfully
Try different prompt variations
Adjust lora_scale if results are too strong/weak
```

### File Access

```
Use 'beam ls flux-lora-data' to list files
Download with 'beam cp beam://flux-lora-data/file.png ./'
Check Beam dashboard for task status
```