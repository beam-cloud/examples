# ComfyUI API Example

An image generation API powered by ComfyUI, running ControlNet workflow

## Deployment

Deploy the app on Beam:

```
beam deploy app.py:handler
```

---

## API Usage

Send a `POST` request to the `/generate` endpoint with the following JSON body:

```json
{
  "prompt": "Your prompt",
  "image_url": "https://your-image-url"
}
```

---

### Example Request:

```json
{
  "prompt": "A photorealistic golden retriever sitting in a field of flowers, soft light, professional lens, background blur",
  "image_url": "https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcRHWNgtJvfMe4yIAOCPTpxRjPalxseEhGfMvW_B3Y3RLjhvwtCIHaYKt0D3K1h2qcWqqvvroakXVGKcAFnTDk7HhA"
}
```

---

### Example Response:

```json
{
  "output_url": " https://app.beam.cloud/output/id/5cc90408-2c40-424f-bb3f-731268e7f100"
}
```
