import gdown
import zipfile
import os
import shutil
import sys
import time

FILE_ID = "1uZiGAX3XCpnjJnEhwmu1Ds8il0dmEbAt"
ZIP_NAME = "Chatbot_Medical_Advice.zip"

def download_with_retry(file_id, output, max_retries=3):
    """Download with retry logic for flaky connections"""
    url = f"https://drive.google.com/uc?id={file_id}"
    
    for attempt in range(max_retries):
        try:
            print(f"⬇️ Downloading model from Google Drive (attempt {attempt + 1}/{max_retries})...")
            gdown.download(url, output, quiet=False)
            
            # Validate download
            if not os.path.exists(output):
                raise FileNotFoundError(f"Download failed - {output} not found")
            
            file_size = os.path.getsize(output)
            if file_size == 0:
                raise ValueError("Downloaded file is empty")
            
            print(f"✅ Download complete. File size: {file_size / (1024*1024):.2f} MB")
            
            # Validate ZIP
            if not zipfile.is_zipfile(output):
                with open(output, 'rb') as f:
                    header = f.read(4)
                    print(f"❌ Downloaded file is not a zip file! Header: {header}")
                raise zipfile.BadZipFile("File is not a valid zip file")
            
            return True
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("🔄 Retrying in 5 seconds...")
                time.sleep(5)
                if os.path.exists(output):
                    os.remove(output)
            else:
                raise

def find_model_directory():
    """Find the final_model directory recursively"""
    print("🔍 Searching for model directory...")
    
    # Common possible locations
    candidates = [
        "final_model",
        "Chatbot_Medical_Advice/final_model",
        "Chatbot_Medical_Advice/Chatbot_Medical_Advice/final_model"
    ]
    
    # Check obvious candidates first
    for path in candidates:
        if os.path.exists(path) and os.path.isdir(path):
            print(f"✅ Found model directory at: {path}")
            return path
    
    # Recursive search
    print("🔍 Performing recursive search...")
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "final_model" or "final_model" in dir_name:
                full_path = os.path.join(root, dir_name)
                print(f"📁 Found candidate: {full_path}")
                return full_path
    
    return None

def extract_and_setup(zip_path):
    """Extract ZIP and set up model directory"""
    print("📦 Extracting archive...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List contents first
            print("📋 ZIP contents:")
            for name in zip_ref.namelist()[:10]:  # Show first 10 files
                print(f"  {name}")
            if len(zip_ref.namelist()) > 10:
                print(f"  ... and {len(zip_ref.namelist()) - 10} more files")
            
            zip_ref.extractall(".")
        
        print("✅ Extraction complete.")
        
    except zipfile.BadZipFile as e:
        print(f"❌ Bad ZIP file: {e}")
        raise
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        raise
    
    # Find and move model directory
    model_path = find_model_directory()
    
    if not model_path:
        print("❌ Could not find final_model directory!")
        print("📁 Current directory structure:")
        for root, dirs, files in os.walk(".", topdown=True):
            level = root.replace(".", "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            if level < 2:  # Limit depth to avoid too much output
                subindent = " " * 2 * (level + 1)
                for file in files[:5]:  # Show first 5 files
                    print(f"{subindent}{file}")
                if len(files) > 5:
                    print(f"{subindent}... and {len(files) - 5} more files")
        
        raise FileNotFoundError("final_model directory not found")
    
    # Move to root if not already there
    if model_path != "final_model":
        dst = "final_model"
        if os.path.exists(dst):
            print("🗑️ Removing existing final_model directory...")
            shutil.rmtree(dst)
        
        print(f"📦 Moving {model_path} to {dst}...")
        shutil.move(model_path, dst)
        print("✅ Model directory moved to root.")
    
    # Verify the setup
    if os.path.exists("final_model"):
        print("✅ final_model directory is ready!")
        model_files = os.listdir("final_model")
        print(f"📄 Model contains {len(model_files)} files/directories")
    else:
        raise FileNotFoundError("final_model setup failed")

def cleanup():
    """Clean up temporary files"""
    files_to_remove = [ZIP_NAME, "Chatbot_Medical_Advice"]
    
    for item in files_to_remove:
        if os.path.exists(item):
            try:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                    print(f"🗑️ Removed directory: {item}")
                else:
                    os.remove(item)
                    print(f"🗑️ Removed file: {item}")
            except Exception as e:
                print(f"⚠️ Could not remove {item}: {e}")

if __name__ == "__main__":
    try:
        # Check available disk space
        if hasattr(shutil, 'disk_usage'):
            total, used, free = shutil.disk_usage(".")
            print(f"💾 Available disk space: {free / (1024**3):.2f} GB")
        
        # Download and setup
        download_with_retry(FILE_ID, ZIP_NAME)
        extract_and_setup(ZIP_NAME)
        cleanup()
        
        print("🎉 Setup completed successfully!")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to clean up on failure
        try:
            cleanup()
        except:
            pass
        
        sys.exit(1)
