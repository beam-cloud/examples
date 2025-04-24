import requests
import base64
from io import BytesIO
from PIL import Image

BEAM_URL = "YOUR_BEAM_URL"  # Replace with your actual Beam Pod URL
API_ENDPOINT = f"{BEAM_URL}/sdapi/v1/img2img"

IMAGE_URL = "https://i.pinimg.com/736x/1a/70/1d/1a701d550cc1bc1d088a7add6cc91278.jpg"
OUTPUT_FILENAME = "upscaled_result.png"

params = {
    "seed": 1337,
    "prompt": "masterpiece, best quality, highres, <lora:more_details:0.5> <lora:SDXLrender_v2.0:1>",
    "dynamic": 6,
    "handfix": "disabled",
    "pattern": False,
    "sharpen": 0,
    "sd_model": "juggernaut_reborn.safetensors",
    "scheduler": "DPM++ 3M SDE Karras",
    "creativity": 0.35,
    "lora_links": "",
    "downscaling": False,
    "resemblance": 0.6,
    "scale_factor": 2,
    "tiling_width": 112,
    "output_format": "png",
    "tiling_height": 144,
    "negative_prompt": "(worst quality, low quality, normal quality:2) JuggernautNegative-neg",
    "num_inference_steps": 18,
    "downscaling_resolution": 768,
}

try:
    image_response = requests.get(IMAGE_URL)
    image = Image.open(BytesIO(image_response.content))
    image = image.convert("RGB")

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    payload = {
        "override_settings": {
            "sd_model_checkpoint": params["sd_model"],
            "sd_vae": "vae-ft-mse-840000-ema-pruned.safetensors",
            "CLIP_stop_at_last_layers": 1,
        },
        "override_settings_restore_afterwards": False,
        "init_images": [img_base64],
        "prompt": params["prompt"],
        "negative_prompt": params["negative_prompt"],
        "steps": params["num_inference_steps"],
        "cfg_scale": params["dynamic"],
        "seed": params["seed"],
        "do_not_save_samples": True,
        "sampler_name": params["scheduler"],
        "denoising_strength": params["creativity"],
        "alwayson_scripts": {
            "Tiled Diffusion": {
                "args": [
                    True,
                    "MultiDiffusion",
                    True,
                    True,
                    1,
                    1,
                    params["tiling_width"],
                    params["tiling_height"],
                    4,
                    8,
                    "4x-UltraSharp",
                    params["scale_factor"],
                    False,
                    0,
                    0.0,
                    3,
                ]
            },
            "Tiled VAE": {
                "args": [
                    True,
                    2048,
                    128,
                    True,
                    True,
                    True,
                    True,
                ]
            },
            "controlnet": {
                "args": [
                    {
                        "enabled": True,
                        "module": "tile_resample",
                        "model": "control_v11f1e_sd15_tile",
                        "weight": params["resemblance"],
                        "image": img_base64,
                        "resize_mode": 1,
                        "lowvram": False,
                        "downsample": 1.0,
                        "guidance_start": 0.0,
                        "guidance_end": 1.0,
                        "control_mode": 1,
                        "pixel_perfect": True,
                        "threshold_a": 1,
                        "threshold_b": 1,
                        "save_detected_map": False,
                        "processor_res": 512,
                    }
                ]
            },
        },
    }

    response = requests.post(API_ENDPOINT, json=payload)
    response.raise_for_status()
    result = response.json()

    upscaled_img_data = base64.b64decode(result["images"][0])
    with open(OUTPUT_FILENAME, "wb") as f:
        f.write(upscaled_img_data)
     
    print(f"Image saved as {OUTPUT_FILENAME}")

except Exception as e:
    print(f"An error occurred: {e}")
