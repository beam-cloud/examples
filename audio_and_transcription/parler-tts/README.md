# Parler TTS API Example

A text-to-speech API service powered by Parler TTS

## Deployment

```bash
beam deploy app.py:generate_speech
```

## API Usage

Send a `POST` request with the following JSON payload:

```json
{
    "prompt": "Your text to convert to speech",
    "description": "Description of the voice/style"
}
```

### Example Request:

```json
{
    "prompt": "On Beam run AI workloads anywhere with zero complexity. One line of Python, global GPUs, full control",
    "description": "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."
}
```

### Example Response:

A generated audio file will be returned. 

## Audio Example

Here’s an example of the generated audio output:

<audio controls>
  <source src="https://app.beam.cloud/output/id/ba83512a-f1b5-4464-a05d-30d6bcdb7cb8" type="audio/mpeg">
</audio>

