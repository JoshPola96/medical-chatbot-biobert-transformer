# setup.py
import os
import requests
import zipfile

def download_and_extract_model():
    file_id = "1uZiGAX3XCpnjJnEhwmu1Ds8il0dmEbAt"  # <-- your Google Drive file ID
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    zip_path = "project_bundle.zip"
    extract_dir = "final_model"

    if os.path.exists(extract_dir):
        print("✅ Model already exists. Skipping download.")
        return

    print("⬇️ Downloading model from Google Drive...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print("📦 Extracting model...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    os.remove(zip_path)
    print("✅ Setup complete.")

if __name__ == "__main__":
    download_and_extract_model()
