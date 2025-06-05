import os
import requests
import urllib.parse
from beam import function, Volume, Image

@function(
  app="volume-imports",
  name="civitai-clone-model",
  secrets=["CIVITAI_API_KEY"],
  memory="4gb",
  image=Image(
    python_packages=["requests"]
  ),
  volumes=[Volume(name="civitai_models", mount_path="/civitai_models")]
)
def handler(*, url: str):
  if not url:
    raise ValueError("url is required")
  
  print(f"Downloading model from {url}")
  model_id = urllib.parse.urlparse(url).path.split("/")[2]
  print(f"Model ID: {model_id}")
  
  try:
    response = requests.get(
      f"https://civitai.com/api/v1/models/{model_id}",
      headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('CIVITAI_API_KEY')}"
      },
    )
    response.raise_for_status()
    data  = response.json()
    
    latest_version = data["modelVersions"][0]
    download_url = latest_version["downloadUrl"]
    print(f"Download URL: {download_url}")
    
    save_path = f"/civitai_models/{data['name'].replace(' ', '_')}"
    with requests.get(download_url, 
                     headers={
                       "Content-Type": "application/json",
                       "Authorization": f"Bearer {os.getenv('CIVITAI_API_KEY')}"
                     },
                     stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        print(f"Total file size: {total_size / (1024*1024):.2f} MB")
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    
    print("Model downloaded successfully")  
    return {
      "model_id": model_id,
      "downloaded_path": save_path
    }
    
  except Exception as e:
    print(f"Failed to get model ID: {str(e)}")
    raise Exception(f"Failed to get model ID: {str(e)}")

if __name__ == "__main__":
  handler(url="https://civitai.com/models/1224788/prefect-illustrious-xl")