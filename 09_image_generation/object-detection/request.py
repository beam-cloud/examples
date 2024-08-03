import requests
import base64

# Beam API details -- make sure to replace with your own credentials
url = 'https://app.beam.cloud/endpoint/id/[ENDPOINT-ID]'
headers = {
    'Connection': 'keep-alive',
    'Content-Type': 'application/json',
    'Authorization': 'Bearer [YOUR-AUTH-TOKEN]'
}

# Load image and encode it to base64
def load_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string

# Send a POST request to the Beam endpoint
def call_beam_api(image_base64):
    data = {
        "image_base64": image_base64
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()


if __name__ == "__main__":
    image_path = "example.jpg"
    image_base64 = load_image_as_base64(image_path)
    result = call_beam_api(image_base64)
    print(result)