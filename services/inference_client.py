import requests


class InferenceClient:
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key
    
    def infer(self, image_path, model_id):
        url = f"{self.api_url}/{model_id}"
        
        with open(image_path, "rb") as f:
            response = requests.post(
                url,
                files={"file": f},
                params={"api_key": self.api_key}
            )
        
        response.raise_for_status()
        return response.json()
