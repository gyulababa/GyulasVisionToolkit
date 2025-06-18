# scr/coreclasses/managers/modelmanager.py

import os
import urllib.request

class ModelManager:
    ULTRALYTICS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"

    def __init__(self, model_dir="models"):
        self.project_root = self._find_project_root()
        self.model_dir = os.path.join(self.project_root, model_dir)
        os.makedirs(self.model_dir, exist_ok=True)

    def load_model(self, model_name):
        """
        model_name: e.g. 'yolov8n.pt', 'yolov8x-seg.pt'
        returns full local path, downloading if missing
        """

        # If full absolute path provided → return directly
        if os.path.isabs(model_name):
            if os.path.exists(model_name):
                print(f"✅ Using absolute model path: {model_name}")
                return model_name
            else:
                raise FileNotFoundError(f"❌ Absolute path not found: {model_name}")

        # Always resolve relative to models folder under project root
        local_path = os.path.join(self.model_dir, model_name)

        if os.path.exists(local_path):
            print(f"✅ Model found locally: {local_path}")
            return local_path

        # If not found — fallback to download logic
        print(f"🔄 Model '{model_name}' not found. Downloading...")

        download_url = f"{self.ULTRALYTICS_URL}/{model_name}"
        try:
            self._download_file(download_url, local_path)
            print(f"\n✅ Download complete: {local_path}")
            return local_path
        except Exception as e:
            print(f"\n❌ Failed to download model: {e}")
            raise

    def _download_file(self, url, dest_path):
        with urllib.request.urlopen(url) as response, open(dest_path, 'wb') as out_file:
            total_length = int(response.headers.get('content-length', 0))
            downloaded = 0
            block_size = 8192
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                self._print_progress(downloaded, total_length)

    def _print_progress(self, downloaded, total_length):
        percent = int(downloaded * 100 / total_length)
        print(f"\r⏬ Downloading: {percent}%", end='')

    def _find_project_root(self):
        """
        Detects project root automatically based on file location.
        """
        this_file = os.path.abspath(__file__)
        core_dir = os.path.dirname(this_file)
        project_root = os.path.abspath(os.path.join(core_dir, ".."))
        return project_root
