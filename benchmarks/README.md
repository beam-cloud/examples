# GPU Benchmark 

## Summary

This report presents benchmarking results for three popular models (Whisper, SDXL, and Llama 3) across four different GPU types (A100-40GB, A10G, RTX 4090, and T4). The benchmarks focus on pure inference performance, measuring average inference time in milliseconds and calculating the resulting requests per second (RPS) that each GPU can handle.

### Benchmark Process
1. Each model was deployed on all four GPU types using Beam 
2. Locust was used to send sequential requests with no concurrency (1 user)
3. Inference time was measured within the model execution code
4. Tests ran for 2 minutes per GPU/model combination
5. Results were aggregated to calculate average inference time and RPS

### Timing Implementation
The implementation tracked pure inference time by measuring only the model execution:

```python
inference_start = time.time()  # Start timing before model execution

# Model execution happens here

inference_end = time.time()    # End timing after model execution
inference_time = (inference_end - inference_start) * 1000 
```

### Testing Code

#### 1. Model Implementation (Whisper Example)
```python
from beam import endpoint, Image, Volume, env
import base64
import requests
from tempfile import NamedTemporaryFile


BEAM_VOLUME_PATH = "./cached_models"


# These packages will be installed in the remote container
if env.is_remote():
    from faster_whisper import WhisperModel, download_model
    import time

# This runs once when the container first starts
def load_models():
    model_path = download_model("large-v3", cache_dir=BEAM_VOLUME_PATH)
    model = WhisperModel(model_path, device="cuda", compute_type="float16")
    return model


@endpoint(
    on_start=load_models,
    name="faster-whisper",
    cpu=12,
    memory="32Gi",
    gpu="A10G", # Changed for each deployment
    image=Image(
        base_image="nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04",
        python_version="python3.10",
    )
    .add_python_packages(
        [
            "git+https://github.com/SYSTRAN/faster-whisper.git",
            "huggingface_hub[hf-transfer]",
            "ctranslate2==4.4.0",
        ]
    )
    .with_envs("HF_HUB_ENABLE_HF_TRANSFER=1"),
    volumes=[
        Volume(
            name="cached_models",
            mount_path=BEAM_VOLUME_PATH,
        )
    ],
)
def transcribe(context, **inputs):
    # Retrieve cached model from on_start
    model = context.on_start_value
    inference_start = time.time()  # Start timing inference

    # Inputs passed to API
    language = inputs.get("language")
    audio_base64 = inputs.get("audio_file")
    url = inputs.get("url")

    if audio_base64 and url:
        return {"error": "Only a base64 audio file OR a URL can be passed to the API."}
    if not audio_base64 and not url:
        return {
            "error": "Please provide either an audio file in base64 string format or a URL."
        }

    binary_data = None

    if audio_base64:
        binary_data = base64.b64decode(audio_base64.encode("utf-8"))
    elif url:
        resp = requests.get(url)
        binary_data = resp.content

    text = ""

    with NamedTemporaryFile() as temp:
        try:
            # Write the audio data to the temporary file
            temp.write(binary_data)
            temp.flush()

            segments, _ = model.transcribe(temp.name, beam_size=5, language=language)

            for segment in segments:
                text += segment.text + " "
             
            inference_end = time.time()  # End timing inference
            inference_time = (inference_end - inference_start) * 1000  # Convert to ms

            print(text)
            return {"text": text, "inference_time_ms": inference_time}

        except Exception as e:
            return {"error": f"Something went wrong: {e}"}
```

#### 2. Locust Load Test Script
```python
from locust import HttpUser, task, events, constant
import csv
import time
import os

class WhisperUser(HttpUser):
    wait_time = constant(1) 
    
    @task
    def test_whisper_inference(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer API_TOKEN"
        }
        
        payload = {
            "url": "https://audio-samples.github.io/samples/mp3/blizzard_unconditional/sample-4.mp3"
        }
        
        start_time = time.time()
        response = self.client.post("/", json=payload, headers=headers)
        end_time = time.time()
    
        total_request_time = (end_time - start_time) * 1000  # Convert to ms
        json_response = response.json()
        inference_time = json_response.get("inference_time_ms")
        
        # Log results
        events.request.fire(
            request_type="POST",
            name="whisper_inference",
            response_time=total_request_time,
            response_length=len(response.content),
            exception=None
        )
        
        # Save inference time
        if inference_time is not None:
            with open("inference_times.csv", "a") as f:
                f.write(f"{inference_time}\n")

@events.init.add_listener
def on_locust_init(environment, **kwargs):
    # Initialize results file
    with open("inference_times.csv", "w") as f:
        f.write("")
    
    print("GPU Benchmark starting...")

@events.quitting.add_listener
def save_results(environment, **kwargs):
    # Calculate average inference time
    inference_times = []
    
    try:
        with open("inference_times.csv", "r") as f:
            for line in f.readlines():
                if line.strip():
                    inference_times.append(float(line.strip()))
    except FileNotFoundError:
        print("No inference times recorded")
        return
    
    if not inference_times:
        print("No valid inference times recorded")
        return
        
    avg_inference_time = sum(inference_times) / len(inference_times)
    rps = 1000 / avg_inference_time if avg_inference_time > 0 else 0
    
    gpu_type = os.environ.get("GPU_TYPE", "A10G")  # Default to A10G if not specified
    
    # Save results
    filename = f"{gpu_type.lower()}_benchmark.csv"
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["GPU", "Avg Inference Time (ms)", "Requests Per Second"])
        writer.writerow([gpu_type, avg_inference_time, rps])
```

#### 3. Run Command
```bash
GPU_TYPE=A10G locust -f locustfile.py --host=https://your-endpoint --headless --run-time 2m --users 1 --spawn-rate 1
```

## Results

### Whisper 
| GPU     | Avg Inference Time (ms) | Requests Per Second |
|---------|------------------------:|--------------------:|
| A100-40GB| 559.36             | 1.79                |
| A10G    | 687.15                  | 1.46                |
| RTX4090 | 732.70                  | 1.36                |
| T4      | 1169.68                 | 0.85                |

### SDXL 
| GPU     | Avg Inference Time (ms) | Requests Per Second |
|---------|------------------------:|--------------------:|
| A100-40GB    | 6218.73                 | 0.16                |
| A10G    | 7552.21                 | 0.13                |
| RTX4090 | 7714.01                 | 0.13                |
| T4      | 12498.96                | 0.08                |

### Llama 3 
| GPU     | Avg Inference Time (ms) | Requests Per Second |
|---------|------------------------:|--------------------:|
| A100-40GB    | 1519.57                 | 0.66                |
| RTX4090 | 1801.05                 | 0.56                |
| A10G    | 13504.73                | 0.07                |
| T4      | 29971.44                | 0.03                |

## Analysis

1. **A100-40GB Performance**: The NVIDIA A100-40GB consistently outperforms other GPUs across all models, particularly for compute-intensive tasks like text generation.

2. **A10G Positioning**: The A10G falls between A100 and RTX 4090 for audio and image tasks, but significantly underperforms for text generation.

3. **T4 Limitations**: The T4 consistently shows the slowest performance, with particularly poor results for language models.

## Conclusion

Based on inference time and throughput measurements:

- **For Speech-to-Text (Whisper)**: All GPUs provide reasonable performance, with A100 only ~30% faster than the A10G.
- **For Image Generation (SDXL)**: The A100 provides modest advantages over A10G and RTX 4090.
- **For Text Generation (Llama 3)**: The A100 shows dramatic performance advantages.

These results can help users select the most appropriate GPU for their specific AI workloads, balancing performance needs against cost considerations.