import requests, zipfile, os, shutil

def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    # handle large files
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    print("✅ Download complete.")

def extract_and_move(zip_path):
    extract_to = "."
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("✅ Extraction complete.")

    # Move the model folder to root if needed
    src = os.path.join("Chatbot_Medical_Advice", "final_model")
    dst = "final_model"
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.move(src, dst)
    print("✅ Moved final_model to root.")

if __name__ == "__main__":
    FILE_ID = "1uZiGAX3XCpnjJnEhwmu1Ds8il0dmEbAt"
    ZIP_NAME = "Chatbot_Medical_Advice.zip"

    print("⬇️ Downloading model from Google Drive...")
    download_file_from_google_drive(FILE_ID, ZIP_NAME)

    print("📦 Extracting model...")
    extract_and_move(ZIP_NAME)

    os.remove(ZIP_NAME)
