import requests
import zipfile
import os

def download_file_from_google_drive(file_id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

# Example usage
file_id = "1uZiGAX3XCpnjJnEhwmu1Ds8il0dmEbAt"
zip_path = "model.zip"

download_file_from_google_drive(file_id, zip_path)

# Unzip safely
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("model")
