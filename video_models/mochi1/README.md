# Parler TTS API Example

A text-to-video API service running Mochi-1

## Deployment

```bash
beam deploy app.py:generate_video
```

## API Usage
Send a POST request with:
```bash
{
    "prompt": "Your prompt for video",
}
```