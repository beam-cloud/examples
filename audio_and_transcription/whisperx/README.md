# Deploying WhisperX: Speech Recognition and Diarization on Beam's Serverless GPU


WhisperX is an advanced speech recognition model that enhances OpenAI's Whisper with features like speaker diarization and word-level timestamp alignment. In this guide, we'll walk through deploying WhisperX on a serverless GPU using Beam. By the end of this tutorial, you'll have a fully functional REST API capable of transcribing and processing audio files with high efficiency.

---

## Overview

We will:

- Set up the cloud GPU environment.
- Define the compute environment and dependencies.
- Load the WhisperX model.
- Create the transcription API.
- Deploy the API.
- Invoke the API to transcribe audio files.

---

## Step 1: Setting Up the Cloud GPU Environment

Beam simplifies the deployment of machine learning models by handling the infrastructure and scaling for you. We'll define our compute environment and specify the dependencies required for WhisperX.

### Defining the Compute Environment

We start by specifying the runtime environment for WhisperX using Beam's `Image` class. This allows us to define the Python packages needed for our application.

```python
# app.py
from beam import endpoint, Image, Volume
import whisperx
from tempfile import NamedTemporaryFile
import base64

volume_path = "./cached_models"

# Define the custom image
image = Image(
    python_packages=["ffmpeg-python", "torch", "torchaudio", "numpy", "whisperx"]
)
```

**Explanation:**

- **`python_packages`**: Lists the Python packages to be installed in the environment. We include essential packages like `torch`, `torchaudio`, and `whisperx`.
- **`volume_path`**: Specifies the path where models will be cached to avoid re-downloading.

---

## Step 2: Loading the WhisperX Model

We use the `on_start` function to load the WhisperX models before serving any requests. This ensures that the models are loaded once and cached in memory, improving performance.

```python
def on_start():
    device = "cuda"  # Use GPU for inference
    model_name = "large-v2"  # Choose the desired model size

    # Load the main WhisperX model
    model = whisperx.load_model(model_name, device, download_root=volume_path)

    # Load the alignment model for word-level timestamps
    alignment_model, metadata = whisperx.load_align_model(
        language_code="en", device=device
    )

    return {
        "model": model,
        "alignment_model": alignment_model,
        "metadata": metadata,
    }
```

**Explanation:**

- **`device = "cuda"`**: Specifies that we want to use the GPU for inference.
- **`model_name`**: Specifies the size of the WhisperX model. Larger models generally provide better accuracy at the cost of increased resource usage.
- **`download_root`**: Uses the specified volume path to cache downloaded models.

---

## Step 3: Creating the Transcription API

We define the API endpoint that will handle audio file uploads, transcribe them using WhisperX, and return the transcribed text.

```python
from beam import endpoint, Volume

@endpoint(
    name="whisperx-deployment",
    image=image,
    cpu=4,
    memory="32Gi",
    gpu="A10G",
    volumes=[
        Volume(
            name="cached_models",
            mount_path=volume_path,
        )
    ],
    on_start=on_start,
)
def transcribe_audio(context: dict, **inputs):
    # Retrieve models from context
    model = context["on_start_value"]["model"]
    alignment_model = context["on_start_value"]["alignment_model"]
    metadata = context["on_start_value"]["metadata"]
    device = "cuda"

    with NamedTemporaryFile(suffix=".wav") as temp_audio:
        # Decode and save the uploaded audio file
        audio_data = base64.b64decode(inputs["audio_file"])
        temp_audio.write(audio_data)
        temp_audio.flush()

        # Step 1: Transcribe with WhisperX
        result = model.transcribe(temp_audio.name, language="en")
        segments = result["segments"]

        # Step 2: Perform word-level alignment
        aligned_segments = whisperx.align(
            segments, alignment_model, metadata, temp_audio.name, device=device
        )

        # Step 3: (Optional) Perform speaker diarization
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=None)
        diarized_segments = diarize_model(
            temp_audio.name, aligned_segments["segments"]
        )

        # Prepare the output text
        transcript = ""
        for segment in diarized_segments["segments"]:
            speaker = segment.get("speaker", "Unknown")
            text = segment["text"]
            transcript += f"[{speaker}]: {text}\n"

    return {"transcript": transcript}
```

**Explanation:**

- **`@endpoint` Decorator**: Registers the function as an API endpoint with Beam.
    - **`name`**: Specifies the name of the deployment.
    - **`image`**: Uses the custom image we defined earlier.
    - **`cpu`, `memory`, `gpu`**: Allocates resources for the endpoint.
    - **`volumes`**: Mounts the volume for cached models.
    - **`on_start`**: Specifies the function to run when the endpoint starts.
- **Function Logic**:
    - **Decoding Audio**: The uploaded audio file is base64-decoded and written to a temporary file.
    - **Transcription**: Uses the WhisperX model to transcribe the audio.
    - **Alignment**: Aligns words to get precise timestamps.
    - **Speaker Diarization**: Optionally labels different speakers in the audio.
    - **Preparing Output**: Constructs the transcript with speaker labels.

---

## Step 4: Deploying the API

With the code in place, you can deploy the API to Beam's serverless infrastructure.

### Deploy the Application

Run the following command to deploy your application:

```bash
beam deploy app.py:transcribe_audio --name whisperx-deployment
```

**Explanation:**

- **`beam deploy`**: Command to deploy the application.
- **`app.py:transcribe_audio`**: Specifies the file and function to deploy.
- **`--name`**: Assigns a name to your deployment.

Beam will handle building the Docker image, setting up the environment, and exposing the API endpoint.

---

## Step 5: Invoking the API

After deployment, you can invoke the API by sending a POST request with the audio file.

### Example Invocation Using Python

```python
# invoke.py
import requests
import json
import base64

# Read and encode the audio file
with open('path/to/your/audio_file.mp3', 'rb') as f:
    audio_data = base64.b64encode(f.read()).decode('utf-8')

# Create the JSON payload
payload = {'audio_file': audio_data}

url = "https://app.beam.cloud/endpoint/whisperx-deployment/v1"
headers = {
    "Authorization": "Bearer YOUR_AUTH_TOKEN",
    "Connection": "keep-alive",
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raises an HTTPError if the response was unsuccessful
    result = response.json()
    print("Response:", result)
except requests.exceptions.RequestException as e:
    print("An error occurred:", e)
```

**Explanation:**

- **Reading the Audio File**: Reads and encodes your audio file in base64.
- **Payload**: Constructs a JSON payload with the encoded audio.
- **Headers**:
    - **`Authorization`**: Replace `YOUR_AUTH_TOKEN` with your actual authentication token provided by Beam.
    - **`Content-Type`**: Specifies that we're sending JSON data.
- **Making the Request**: Sends a POST request to the API endpoint.
- **Handling the Response**: Checks for errors and prints the transcription result.

**Important Note:** Remember to keep your authentication token secure and not share it publicly.

---

## Conclusion

You've successfully deployed WhisperX as a serverless speech recognition API using Beam. Your API can now handle:

- **Transcription**: Converting audio to text.
- **Word Alignment**: Providing precise timestamps for each word.
- **Speaker Diarization**: Identifying and labeling different speakers in the audio.

**Benefits of Using Beam:**

- **Scalability**: Automatically scales with your workload.
- **Ease of Deployment**: Simplifies the deployment process with minimal configuration.
- **Resource Management**: Handles infrastructure and resource allocation for you.

---

## Next Steps

- **Customize the Model**: Adjust the `model_name` in the `on_start` function to use different model sizes based on your needs.
- **Error Handling**: Enhance the API to handle different types of errors and edge cases.
- **Additional Features**: Integrate language detection, punctuation restoration, or other NLP tasks.

---

## Resources

- **Beam Documentation**: [https://docs.beam.cloud](https://docs.beam.cloud)
- **WhisperX GitHub Repository**: [https://github.com/m-bain/whisperX](https://github.com/m-bain/whisperX)
- **OpenAI Whisper**: [https://openai.com/blog/whisper/](https://openai.com/blog/whisper/)

---

**Deploy to production in minutes with Beam.**

Beam streamlines the deployment of machine learning models, allowing you to focus on building and scaling your applications. With just a few lines of code, you have a production-ready API leveraging powerful GPU resources.

---

**Feel free to reach out if you have any questions or need further assistance with your deployment!**

