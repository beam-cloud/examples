import requests

BEAM_TOKEN = "YOUR_BEAM_TOKEN"
BEAM_URL = "YOUR_BEAM_APP_URL"


class BeamService:
    def __init__(self, prompt):
        self.url = BEAM_URL
        self.headers = {
            "Authorization": f"Bearer {BEAM_TOKEN}",
            "Content-Type": "application/json",
        }
        self.data = {"prompt": prompt}

    def call_api(self):
        response = requests.post(
            self.url, headers=self.headers, json=self.data, stream=True
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Request failed with status code {response.status_code}")
