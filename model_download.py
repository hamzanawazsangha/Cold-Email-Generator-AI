import gdown
import zipfile
import os

# Google Drive file ID (extract it from the sharing link)
file_id = "1hoR_G1-41p5_ZCfxzOKy49mCLptZPNdV"

# File names
zip_file = "model.zip"  # The ZIP file name
extract_folder = "models"  # Folder to extract the model

# Download the ZIP file from Google Drive
gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_file, quiet=False)

# Extract the ZIP file
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# Remove the ZIP file after extraction
os.remove(zip_file)

print(f"Model downloaded and extracted to '{extract_folder}'")
