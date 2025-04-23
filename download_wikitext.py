import os
import urllib.request
import zipfile
import shutil

def download_wikitext2():
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    wikitext_dir = os.path.join(data_dir, 'wikitext-2')
    os.makedirs(wikitext_dir, exist_ok=True)
    
    # Define file paths
    zip_path = os.path.join(data_dir, 'wikitext-2-v1.zip')
    
    # Check if files already exist
    train_path = os.path.join(wikitext_dir, 'wiki.train.tokens')
    if os.path.exists(train_path):
        print(f"WikiText-2 dataset already exists at {wikitext_dir}")
        return wikitext_dir
    
    # URL for WikiText-2 dataset
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
    
    # Download the zip file
    print("Downloading WikiText-2 dataset...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete.")
    
    # Extract the files
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all files to a temporary directory
        temp_dir = os.path.join(data_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        zip_ref.extractall(temp_dir)
        
        # Move the files to the wikitext-2 directory
        wikitext_extracted = os.path.join(temp_dir, 'wikitext-2')
        for file in os.listdir(wikitext_extracted):
            src = os.path.join(wikitext_extracted, file)
            dst = os.path.join(wikitext_dir, file)
            shutil.move(src, dst)
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
    
    # Clean up zip file
    os.remove(zip_path)
    
    print(f"WikiText-2 dataset extracted to {wikitext_dir}")
    return wikitext_dir

if __name__ == "__main__":
    download_wikitext2()
