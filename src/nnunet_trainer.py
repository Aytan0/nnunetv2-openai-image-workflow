"""
nnUNet Trainer - Conda Environment Compatible

This module handles the training of nnUNet models using the prepared datasets.
It provides functionality to run the training process with GPU support and
monitor the training progress. This version is designed to work with conda environments
and includes robust error handling.
"""

import os
import sys
import time
import logging
import subprocess
import platform
from pathlib import Path
from dotenv import load_dotenv

from dotenv import load_dotenv

# Configure logging with UTF-8 encoding
def setup_logging():
    """Set up logging with file and console handlers."""
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Create a formatter that handles encoding issues
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create file handler with UTF-8 encoding
    file_handler = logging.FileHandler(
        str(log_dir / 'nnunet_trainer.log'), 
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # Create console handler with encoding error handling
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure the logger
    logger = logging.getLogger('nnunet_trainer')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

class NNUNetTrainer:
    """Class to handle nnUNet training with conda environment compatibility."""
    
    def __init__(self, nnunet_data_dir=None, conda_env="nnunetv2"):
        """
        Initialize the NNUNetTrainer.
        
        Args:
            nnunet_data_dir (str, optional): Directory for nnUNet data.
                                            Defaults to '../data/nnUNet'.
            conda_env (str, optional): Name of conda environment with nnUNetV2.
                                      Defaults to "nnunetv2".
        """
        # Load environment variables
        load_dotenv()
        
        # Store conda environment name
        self.conda_env = conda_env
        logger.info(f"Using conda environment: {self.conda_env}")
        
        # Detect operating system
        self.is_windows = platform.system() == 'Windows'
        logger.info(f"Operating system: {platform.system()}")
        
        # Set directories using Path for cross-platform compatibility
        base_dir = Path(__file__).parent.parent.absolute()
        
        if nnunet_data_dir is None:
            self.nnunet_data_dir = base_dir / 'data' / 'nnUNet'
        else:
            self.nnunet_data_dir = Path(nnunet_data_dir).absolute()
        
        # Create nnUNet directories
        raw_dir = self.nnunet_data_dir / 'nnUNet_raw'
        preprocessed_dir = self.nnunet_data_dir / 'nnUNet_preprocessed'
        results_dir = self.nnunet_data_dir / 'nnUNet_results'
        
        # Create directories if they don't exist
        raw_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set nnUNet environment variables with proper path formatting
        os.environ['nnUNet_raw'] = str(raw_dir)
        os.environ['nnUNet_preprocessed'] = str(preprocessed_dir)
        os.environ['nnUNet_results'] = str(results_dir)
        
        logger.info(f"nnUNet data directory: {self.nnunet_data_dir}")
        logger.info(f"nnUNet_raw: {os.environ['nnUNet_raw']}")
        logger.info(f"nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
        logger.info(f"nnUNet_results: {os.environ['nnUNet_results']}")
        
        # Check if conda environment is available
        self._check_conda_environment()
    
    def _check_conda_environment(self):
        """
        Check if conda environment with nnUNetV2 is available.
        
        Returns:
            bool: True if conda environment is available, False otherwise.
        """
        try:
            # Check if conda is available
            result = self._run_command(
                ["conda", "info", "--envs"],
                "Checking conda environments"
            )
            
            if not result:
                logger.error("ERROR: Conda is not available or not in PATH")
                return False
            
            # Check if the specified conda environment exists
            result = self._run_command(
                ["conda", "env", "list"],
                "Listing conda environments"
            )
            
            if not result:
                logger.error(f"ERROR: Failed to list conda environments")
                return False
            
            logger.info(f"Conda environment check completed")
            return True
        except Exception as e:
            logger.error(f"ERROR: Failed to check conda environment: {str(e)}")
            return False
    
    def _run_command(self, cmd, description, shell=False, env=None):
        """
        Run a command with proper error handling and logging.
        
        Args:
            cmd (list): Command to run.
            description (str): Description of the command for logging.
            shell (bool, optional): Whether to run command in shell. Defaults to False.
            env (dict, optional): Environment variables. Defaults to None.
            
        Returns:
            bool: True if command successful, False otherwise.
        """
        try:
            # Log the command
            cmd_str = " ".join(str(c) for c in cmd) if isinstance(cmd, list) else cmd
            logger.info(f"Running {description}: {cmd_str}")
            
            # Prepare environment
            if env is None:
                env = os.environ.copy()
            
            # Run the command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                shell=shell,
                env=env,
                bufsize=1,
                encoding='utf-8',
                errors='replace'  # Handle encoding errors gracefully
            )
            
            # Collect output and error
            stdout_lines = []
            stderr_lines = []
            
            # Log output in real-time
            for line in process.stdout: # type: ignore
                line = line.strip()
                stdout_lines.append(line)
                logger.info(f"[{description}] {line}")
            
            # Wait for process to complete
            process.wait()
            
            # Collect any remaining stderr
            for line in process.stderr: # type: ignore
                line = line.strip()
                stderr_lines.append(line)
                logger.error(f"[{description}] ERROR: {line}")
            
            # Check if command was successful
            if process.returncode == 0:
                logger.info(f"{description} completed successfully")
                return True
            else:
                logger.error(f"ERROR: {description} failed with return code {process.returncode}")
                if stderr_lines:
                    logger.error(f"Error details: {' '.join(stderr_lines)}")
                return False
                
        except Exception as e:
            logger.error(f"ERROR: Exception during {description}: {str(e)}")
            return False
    
    def _run_conda_command(self, cmd, description):
        """
        Run a command in the conda environment.
        
        Args:
            cmd (list): Command to run in conda environment.
            description (str): Description of the command for logging.
            
        Returns:
            bool: True if command successful, False otherwise.
        """
        try:
            # Construct the conda run command
            conda_cmd = ["conda", "run", "-n", self.conda_env] + cmd
            
            # Run the command
            return self._run_command(conda_cmd, description)
                
        except Exception as e:
            logger.error(f"ERROR: Failed to run conda command: {str(e)}")
            return False
    
    def preprocess_dataset(self, task_id):
        """
        Preprocess the dataset for training.
        
        Args:
            task_id (int): ID of the task to preprocess.
            
        Returns:
            bool: True if preprocessing successful, False otherwise.
        """
        try:
            logger.info(f"Preprocessing dataset for Task {task_id}...")
            
            # Build the command for conda environment
            cmd = [
                "nnUNetv2_plan_and_preprocess",
                "--dataset_id", str(task_id),
                "--configuration", "3d_fullres",
                "--num_processes", "8"
            ]
            
            # Run the command in conda environment
            return self._run_conda_command(cmd, f"Preprocessing dataset for Task {task_id}")
                
        except Exception as e:
            logger.error(f"ERROR: Failed to preprocess dataset: {str(e)}")
            return False
    
    def train(self, task_id, task_name, fold=0, trainer="nnUNetTrainer", device="cuda"):
        """
        Train nnUNet model.
        
        Args:
            task_id (int): ID of the task to train.
            task_name (str): Name of the task to train.
            fold (int, optional): Fold to train. Defaults to 0.
            trainer (str, optional): Trainer class to use. Defaults to "nnUNetTrainer".
            device (str, optional): Device to use for training. Defaults to "cuda".
            
        Returns:
            bool: True if training successful, False otherwise.
        """
        try:
            logger.info(f"Training nnUNet model for Task {task_id}, fold {fold}...")
            
            # Build the command for conda environment
            dataset_name = f"Dataset{task_id:03d}_{task_name}"
            cmd = [
                "nnUNetv2_train",
                dataset_name,
                "3d_fullres",
                str(fold),
                "--trainer", trainer,
                "--device", device
            ]
            
            # Run the command in conda environment
            return self._run_conda_command(cmd, f"Training model for Task {task_id}, fold {fold}")
                
        except Exception as e:
            logger.error(f"ERROR: Failed to train model: {str(e)}")
            return False
    
    def predict(self, task_id, task_name, input_folder, output_folder, folds="all", device="cuda"):
        """
        Run inference with the trained model.
        
        Args:
            task_id (int): ID of the task.
            task_name (str): Name of the task.
            input_folder (str): Folder containing input images.
            output_folder (str): Folder to save predictions.
            folds (str, optional): Folds to use for prediction. Defaults to "all".
            device (str, optional): Device to use for prediction. Defaults to "cuda".
            
        Returns:
            bool: True if prediction successful, False otherwise.
        """
        try:
            logger.info(f"Running prediction for Task {task_id}...")
            
            # Convert paths to absolute paths
            input_folder = str(Path(input_folder).absolute())
            output_folder = str(Path(output_folder).absolute())
            
            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            dataset_name = f"Dataset{task_id:03d}_{task_name}"
            # Build the command for conda environment
            cmd = [
                "nnUNetv2_predict",
                "-i", input_folder,
                "-o", output_folder,
                "-d", dataset_name,
                "-c", "3d_fullres",
                "-f", folds,
                "--device", device
            ]
            
            # Run the command in conda environment
            return self._run_conda_command(cmd, f"Prediction for Task {task_id}")
                
        except Exception as e:
            logger.error(f"ERROR: Failed to run prediction: {str(e)}")
            return False

def main():
    """Main function to demonstrate usage."""
    # Create trainer instance
    trainer = NNUNetTrainer()
    
    # Task ID
    task_id = 501  # Same as used in dataset_converter.py
    task_name = "LiverVesselSeg"
    
    # Preprocess dataset
    logger.info("Step 1: Preprocessing dataset")
    if trainer.preprocess_dataset(task_id):
        # Train model (fold 0)
        logger.info("Step 2: Training model")
        if trainer.train(task_id, task_name, fold=0):
            logger.info(f"Training completed for Task {task_id}, fold 0")
            
            # Example: Run prediction on test data
            logger.info("Step 4: Running prediction")
            test_folder = Path(os.environ['nnUNet_raw']) / f"Dataset{task_id:03d}_{task_name}" / "imagesTs"
            output_folder = Path(__file__).parent.parent / "data" / "predictions"
            
            if test_folder.exists() and any(test_folder.iterdir()):
                if trainer.predict(task_id, task_name, str(test_folder), str(output_folder)):
                    logger.info(f"Prediction completed for Task {task_id}")
                else:
                    logger.error(f"Prediction failed for Task {task_id}")
                    logger.error("ERROR CODE: PREDICTION_FAILED")
            else:
                logger.warning(f"Test folder {test_folder} does not exist or is empty. Skipping prediction.")
                logger.warning("ERROR CODE: TEST_FOLDER_MISSING")
        else:
            logger.error(f"Training failed for Task {task_id}")
            logger.error("ERROR CODE: TRAINING_FAILED")
    else:
        logger.error(f"Preprocessing failed for Task {task_id}")
        logger.error("ERROR CODE: PREPROCESSING_FAILED")

if __name__ == "__main__":
    main()
