import gdown
import zipfile
import os
import shutil

FILE_ID = "1uZiGAX3XCpnjJnEhwmu1Ds8il0dmEbAt"
ZIP_NAME = "Chatbot_Medical_Advice.zip"

def download_zip_with_gdown(file_id, output):
    url = f"https://drive.google.com/uc?id={file_id}"
    print("⬇️ Downloading model from Google Drive...")
    gdown.download(url, output, quiet=False)

    # Validate ZIP
    if not zipfile.is_zipfile(output):
        with open(output, 'rb') as f:
            print("❌ Downloaded file is not a zip file!")
            print("Downloaded file starts with:", f.read(4))
        raise zipfile.BadZipFile("File is not a valid zip file")
    
    print("✅ Download complete.")

def extract_and_move(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")

    print("📦 Extracted model.")

    # Move final_model from extracted folder to root
    src = os.path.join("Chatbot_Medical_Advice", "final_model")
    dst = "final_model"

    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.move(src, dst)
    print("✅ Moved 'final_model' to root directory.")

if __name__ == "__main__":
    try:
        download_zip_with_gdown(FILE_ID, ZIP_NAME)
        extract_and_move(ZIP_NAME)
        os.remove(ZIP_NAME)
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        exit(1)
