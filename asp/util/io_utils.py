import os
import gdown


def download_weights(file_id, weights_path):
    if os.path.exists(weights_path):
        print(f"Weights already exist at {weights_path}.")
        return

    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, weights_path, quiet=False)