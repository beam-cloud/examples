# app.py
# Create the Beam WhisperX API Endpoint

from beam import endpoint, Image, Volume
import whisperx
from tempfile import NamedTemporaryFile
import base64

volume_path = "./cached_models"

# Define the custom image
image = (
    Image(
        python_packages=["ffmpeg-python", "torch", "torchaudio", "numpy", "whisperx"]
    )
)

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
    # Function implementation
    with NamedTemporaryFile(suffix=".wav") as temp_audio:
        # Decode and save the uploaded audio file
        audio_data = base64.b64decode(inputs["audio_file"])
        temp_audio.write(audio_data)
        temp_audio.flush()

    # Step 1: Transcribe with WhisperX
    result = model.transcribe(temp_audio.name, language=language)
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
