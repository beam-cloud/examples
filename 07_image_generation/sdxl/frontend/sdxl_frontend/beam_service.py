import requests


class BeamService:
    def __init__(self, prompt):
        self.url = (
            "https://app.beam.cloud/endpoint/id/93c1a2b2-d61c-4264-bcfe-999c6bf78117"
        )
        self.headers = {
            "Authorization": "Bearer FchYCvVx4vkt1oBJkxyStLqAgT-nedRQqRojzisXU1NEijUH1_ih-IpJegWylyj-6kFD_UDnfdGPiYdTKMatrQ==",
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
