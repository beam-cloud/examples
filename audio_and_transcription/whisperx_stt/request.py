import requests

AUTH_TOKEN = "BEAM_AUTH_TOKEN"
BEAM_URL = (
    "BEAM_URL"  # Will look something like: id/618b1458-0a84-4be5-ae8f-7d70e76374d9
)
AUDIO_URL = (
    "https://audio-samples.github.io/samples/mp3/blizzard_unconditional/sample-4.mp3"
)

payload = {"url": AUDIO_URL}

url = f"https://app.beam.cloud/endpoint/{BEAM_URL}"
headers = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
}

try:
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    print("Response:", result)
except requests.exceptions.RequestException as e:
    print("An error occurred:", e)
