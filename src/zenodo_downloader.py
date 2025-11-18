import os
import zipfile
import requests
import logging
import traceback

logger = logging.getLogger("zenodo_downloader")

class ZenodoDownloader:
    def __init__(self, dataset_manager):
        self.raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data", "raw")
        os.makedirs(self.raw_data_dir, exist_ok=True)
        self.dataset_manager = dataset_manager

    def download_dataset(self, record_id: str) -> bool:
        logger.info(f"Zenodo dataset download initiated for record ID: {record_id}")
        zenodo_url = f"https://zenodo.org/api/records/{record_id}"
        try:
            response = requests.get(zenodo_url)
            response.raise_for_status()
            record_data = response.json()

            files = record_data.get("files", [])
            if not files:
                logger.error(f"No files found for Zenodo record ID: {record_id}")
                return False

            # Find the largest file, assuming it's the main dataset archive
            main_file = None
            for f in files:
                if main_file is None or f["size"] > main_file["size"]:
                    main_file = f
            
            if not main_file:
                logger.error(f"No suitable file found for download in Zenodo record ID: {record_id}")
                return False

            download_url = main_file["links"]["self"]
            file_name = main_file["key"]
            
            # Create a unique directory for the dataset
            next_id = self.dataset_manager.get_next_dataset_id()
            dataset_name = f"dataset_{next_id:03d}"
            dataset_path = os.path.join(self.raw_data_dir, dataset_name)
            os.makedirs(dataset_path, exist_ok=True)
            download_path = os.path.join(dataset_path, file_name)

            logger.info(f"Downloading {file_name} from {download_url} to {download_path}")
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                with open(download_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logger.info(f"Download complete: {download_path}")

            if file_name.endswith(".zip"):
                logger.info(f"Extracting {file_name} to {dataset_path}")
                with zipfile.ZipFile(download_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_path)
                logger.info(f"Extraction complete: {dataset_path}")
                os.remove(download_path) # Remove the zip file after extraction

            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading from Zenodo: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during Zenodo download: {e}")
            return False


