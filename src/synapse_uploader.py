import os
import logging
import synapseclient

logger = logging.getLogger(__name__)

class SynapseUploader:
    def __init__(self):
        self.syn = synapseclient.Synapse()
        try:
            self.syn.login()
            logger.info("Successfully logged into Synapse.")
        except Exception as e:
            logger.error(f"Failed to log into Synapse: {e}")
            raise

    def upload_results(self, project_id, results_path):
        try:
            # Example: Upload a file to a Synapse project
            # Replace with actual logic to upload nnUNet results
            entity = synapseclient.File(results_path, parentId=project_id)
            entity = self.syn.store(entity)
            logger.info(f"Successfully uploaded {results_path} to Synapse project {project_id} with ID {entity.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload results to Synapse: {e}")
            return False


