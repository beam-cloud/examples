import requests

AUTH_TOKEN = "YOUR_BEAM_AUTH_TOKEN"  # Replace with your actual Beam auth token

url = "https://78f2e172-6ae5-4cef-b43a-9823238f.app.beam.cloud"

headers = {
    "Connection": "keep-alive",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AUTH_TOKEN}",
}

reasoning_trace = """
    Product ID: P-001
    Product Name: Wireless Ergonomic Mouse
    Price: $29.99
    In Stock: Yes
    Features: Adjustable DPI, Rechargeable Battery, Silent Clicks
    Compatible OS: Windows, macOS, Linux
"""

data = {
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant that understands and translates text to JSON format",
        },
        {"role": "user", "content": reasoning_trace},
    ]
}

response = requests.post(url, headers=headers, json=data)
print(response.text)
