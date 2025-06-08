import requests
import zipfile
import os
import shutil

def download_file_from_google_drive(file_id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        with open(destination, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    print("⬇️ Downloading model from Google Drive...")

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

    # Confirm it's a ZIP file
    if not zipfile.is_zipfile(destination):
        print("❌ Downloaded file is not a zip file!")
        with open(destination, 'rb') as f:
            print("Downloaded file starts with:", f.read(4))
        raise zipfile.BadZipFile("File is not a valid zip file")

    print("✅ Download complete.")

def extract_and_move(zip_path):
    extract_to = "."
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("✅ Extraction complete.")

    # Move final_model to root
    src = os.path.join("Chatbot_Medical_Advice", "final_model")
    dst = "final_model"
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.move(src, dst)
    print("✅ Moved final_model to root.")

if __name__ == "__main__":
    FILE_ID = "1uZiGAX3XCpnjJnEhwmu1Ds8il0dmEbAt"
    ZIP_NAME = "Chatbot_Medical_Advice.zip"

    try:
        download_file_from_google_drive(FILE_ID, ZIP_NAME)
        extract_and_move(ZIP_NAME)
        os.remove(ZIP_NAME)
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        exit(1)
