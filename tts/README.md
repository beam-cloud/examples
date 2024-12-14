# Parler TTS API Example

A text-to-speech API service powered by Parler TTS

## Deployment

```bash
beam deploy app.py:generate_speech
```

## API Usage
Send a POST request with:
```bash
{
    "prompt": "Your text to convert to speech",
    "description": "Description of the voice/style"
}
```