# vLLM API Example

This example demonstrates how to run a vLLM-powered API server that's compatible with OpenAI's API format using the InternVL2 5-8B model.

1. Deploy the API server:

```bash
beam deploy app.py:generate
```

## API Usage

Make sure you have OpenAI SDK installed if not ```pip install openai``

```python
import base64
import requests
from openai import OpenAI

openai_api_key = "your-beam-token"
openai_api_base = "https:your-beam-app/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

image_url = "https://tinypng.com/static/images/boat.png"

chat_completion_from_url = client.chat.completions.create(
        messages=[{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    },
                },
            ],
        }],
        model="InternVL2_5-8B"
    )
print(chat_completion_from_url)
```