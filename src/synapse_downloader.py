"""
Synapse.org Dataset Downloader


"""


from dotenv import load_dotenv
import os
import time
import logging
import subprocess
import synapseclient
import synapseutils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'synapse_downloader.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('synapse_downloader')

# Load environment variables



class SynapseDownloader:
    """Class to handle Synapse.org authentication and dataset downloading."""

    def __init__(self, download_dir=None):
        """
        Initialize the SynapseDownloader.

        Args:
            download_dir (str, optional): Directory to save downloaded files.
                                          Defaults to '../data/raw'.
        """
        self.syn = synapseclient.Synapse()
        self.download_dir = download_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
        os.makedirs(self.download_dir, exist_ok=True)
        self.synapse_token = os.getenv("SYNAPSE_TOKEN") or input("Lutfen Synapse tokeninizi girin: ")

    def login(self):
        """
        Log in to Synapse.org using token from .env file.

        Returns:
            bool: True if login successful, False otherwise.
        """
        try:
            logger.info("Attempting to log in to Synapse.org using token...")
            if not self.synapse_token:
                logger.error("ERROR: SYNAPSE_TOKEN not found in .env file.")
                return False

            self.syn.login(authToken=self.synapse_token)
            logger.info("Successfully logged in to Synapse.org using token.")
            return True

        except Exception as e:
            logger.error(f"ERROR: Login failed: {str(e)}")
            return False

    def download_dataset(self, synapse_project_id):
        """
        Download dataset from the specified Synapse project ID using synapseutils.

        Args:
            synapse_project_id (str): The Synapse Project ID to download from.

        Returns:
            bool: True if download successful, False otherwise.
        """
        try:
            logger.info("Starting dataset download from Synapse.org using synapseutils...")

            if not self.login():
                logger.error("ERROR: Login failed.")
                return False

            logger.info(f"Downloading dataset from Synapse project ID: {synapse_project_id} to {self.download_dir}")
            files = synapseutils.syncFromSynapse(self.syn, synapse_project_id, path=self.download_dir)

            if files:
                logger.info("Dataset download completed successfully using synapseutils.")
                return True
            else:
                logger.error("ERROR: Dataset download failed or no files were downloaded.")
                return False

        except Exception as e:
            logger.error(f"ERROR: Failed to download dataset: {str(e)}")
            return False

    def close(self):
        """
        Placeholder for closing resources (no action needed with synapseclient).
        """
        pass


