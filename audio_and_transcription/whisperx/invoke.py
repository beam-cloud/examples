# invoke.py
# Run this script to make an API request to the endpoint we created

import requests
import json
import base64

# Read and encode the audio file
with open('/path/to/test_file.mp3', 'rb') as f:
    audio_data = base64.b64encode(f.read()).decode('utf-8')

# Create the JSON payload
payload = {'audio_file': audio_data}

url = "https://app.beam.cloud/endpoint/your/url"
headers = {
    "Authorization": "Bearer ...",
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
