"""
Workflow Manager for nnUNet Automation
Handles the complete pipeline from data download to visualization
"""

import os
import sys
import logging
import traceback
import subprocess
import shutil
import json
from pathlib import Path
from dotenv import load_dotenv

from src.dataset_converter import DatasetConverter
from src.nnunet_trainer import NNUNetTrainer
from src.visualizer import Visualizer
from src.synapse_downloader import SynapseDownloader
from src.synapse_uploader import SynapseUploader
from src.zenodo_downloader import ZenodoDownloader
from src.ai_organizer import AIOrganizer
from src.dataset_manager import DatasetManager

logger = logging.getLogger("workflow")

class WorkflowManager:
    """Main workflow manager for nnUNet automation."""
    
    def __init__(self):
        """Initialize workflow manager with required paths."""
        self.CONDA_ENV = "nnunetv2"
        
        base_dir = Path(__file__).parent.parent
        data_dir = base_dir / "data"
        
        self.nnunet_raw = str(data_dir / "nnUNet" / "nnUNet_raw")
        self.nnunet_preprocessed = str(data_dir / "nnUNet" / "nnUNet_preprocessed")
        self.nnunet_results = str(data_dir / "nnUNet" / "nnUNet_results")
        self.predictions_dir = str(data_dir / "predictions")
        self.raw_data_dir = str(data_dir / "raw")
        self.downloads_dir = str(base_dir / "downloads")
        
        self.dataset_manager = DatasetManager(self.nnunet_raw)
        
        self.setup_environment()

    def setup_environment(self):
        """Set up environment variables and create necessary directories."""
        try:
            logger.info("Setting up environment...")
            
            for directory in [
                self.nnunet_raw,
                self.nnunet_preprocessed,
                self.nnunet_results,
                self.predictions_dir,
                self.raw_data_dir,
                self.downloads_dir
            ]:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Directory ensured: {directory}")

            os.environ['nnUNet_raw'] = self.nnunet_raw
            os.environ['nnUNet_preprocessed'] = self.nnunet_preprocessed
            os.environ['nnUNet_results'] = self.nnunet_results
            os.environ['nnUNet_n_proc_DA'] = "12"

            logger.info(f"nnUNet_raw: {os.environ['nnUNet_raw']}")
            logger.info(f"nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
            logger.info(f"nnUNet_results: {os.environ['nnUNet_results']}")
            logger.info("Environment setup completed successfully")
            
            return True
        except Exception as e:
            logger.error(f"Failed to set up environment: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def run_conda_command(self, cmd, description):
        """Run command in conda environment."""
        try:
            logger.info(f"Running {description}...")
            
            conda_cmd = ["conda", "run", "-n", self.CONDA_ENV] + cmd
            logger.info(f"Command: {' '.join(conda_cmd)}")
            
            process = subprocess.Popen(
                conda_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace'
            )
            
            stdout_lines = []
            for line in process.stdout:
                line = line.strip()
                if line:
                    stdout_lines.append(line)
                    logger.info(f"[{description}] {line}")
            
            process.wait()
            
            stderr_lines = []
            for line in process.stderr:
                line = line.strip()
                if line:
                    stderr_lines.append(line)
                    logger.error(f"[{description}] ERROR: {line}")
            
            if process.returncode == 0:
                logger.info(f"{description} completed successfully")
                return True
            else:
                logger.error(f"{description} failed with return code {process.returncode}")
                if stderr_lines:
                    logger.error(f"Error details: {' '.join(stderr_lines)}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to run conda command: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def convert_datasets(self, args=None, input_folder=None):
        """Convert datasets to nnUNet format."""
        try:
            logger.info("Starting dataset conversion...")
            
            if args is None:
                return self.convert_datasets_auto()
            
            converter = DatasetConverter()
            
            if input_folder:
                use_input_folder = input_folder
            elif hasattr(args, 'input_folder') and args.input_folder:
                use_input_folder = args.input_folder
            else:
                use_input_folder = self.raw_data_dir
            
            if hasattr(args, 'task_name') and args.task_name:
                task_name = args.task_name
            else:
                task_name, task_id = self.dataset_manager.create_dataset_name()
                args.task_name = task_name
                args.task_id = task_id
                logger.info(f"Auto-generated dataset name: {task_name} (ID: {task_id})")
            
            if hasattr(args, 'task_id') and args.task_id:
                task_id = args.task_id
            else:
                task_id = self.dataset_manager.get_next_dataset_id()
                args.task_id = task_id
                logger.info(f"Auto-generated dataset ID: {task_id}")
            
            if converter.organize_dataset(use_input_folder, task_name, task_id):
                logger.info(f"Dataset successfully converted: Dataset{task_id:03d}_{task_name}")
                return True
            else:
                logger.error("Failed to convert dataset")
                return False
                
        except Exception as e:
            logger.error(f"Dataset conversion error: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def convert_datasets_auto(self) -> bool:
        """Automatically convert datasets with AI organization."""
        try:
            logger.info("Starting automatic dataset conversion...")
            
            ai_organizer = AIOrganizer(self.nnunet_raw)
            return ai_organizer.organize_zip_with_ai()
            
        except Exception as e:
            logger.error(f"Automatic dataset conversion error: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def _run_preprocessing(self, dataset_id: int) -> bool:
        """Run nnUNet preprocessing for a dataset."""
        try:
            print("\n" + "="*60)
            print("Running Dataset Preprocessing")
            print("="*60)
            print("\nThis process will:")
            print("  1. Analyze dataset properties")
            print("  2. Calculate intensity statistics")
            print("  3. Create training plans")
            print("  4. Preprocess and resample all images")
            print("\nEstimated time: 10-30 minutes")
            print("\nPlease wait...\n")
            
            cmd = [
                "nnUNetv2_plan_and_preprocess",
                "-d", str(dataset_id),
                "-np", "1",
                "--verify_dataset_integrity"
            ]
            
            logger.info(f"Running preprocessing: {' '.join(cmd)}")
            print(f"Command: {' '.join(cmd)}\n")
            print("Using single-process mode for better error handling\n")
            print("="*60)
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                universal_newlines=True
            )
            
            for line in process.stdout:
                print(line, end='')
                logger.info(line.strip())
            
            process.wait()
            
            if process.returncode == 0:
                print("\n" + "="*60)
                print("✓ Preprocessing Completed Successfully!")
                print("="*60)
                
                preprocessed_dirs = list(Path(self.nnunet_preprocessed).glob(f"Dataset{dataset_id:03d}_*"))
                
                if preprocessed_dirs:
                    print(f"\nPreprocessed files saved to:")
                    print(f"  {preprocessed_dirs[0]}")
                    logger.info(f"Preprocessing files verified at: {preprocessed_dirs[0]}")
                
                logger.info("Preprocessing completed successfully")
                return True
            else:
                print("\n" + "="*60)
                print("✗ Preprocessing Failed")
                print("="*60)
                print("\nImage-label dimension mismatches detected.")
                print("\nOptions to fix:")
                print("  1. Remove mismatched files manually")
                print("  2. Re-run data conversion with proper alignment")
                print("  3. Skip verification: remove --verify_dataset_integrity")
                
                logger.error(f"Preprocessing failed with return code: {process.returncode}")
                
                retry = input("\nRetry without integrity check? [y/N]: ").strip().lower()
                if retry == 'y':
                    return self._run_preprocessing_no_verify(dataset_id)
                
                return False
                
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"\nPreprocessing error: {str(e)}")
            return False

    def _run_preprocessing_no_verify(self, dataset_id: int) -> bool:
        """Run preprocessing without dataset integrity verification."""
        try:
            print("\n" + "="*60)
            print("Running Preprocessing (No Integrity Check)")
            print("="*60)
            print("\n⚠️  Skipping dimension verification")
            print("   nnUNet will attempt to resample all files automatically\n")
            
            cmd = [
                "nnUNetv2_plan_and_preprocess",
                "-d", str(dataset_id),
                "-np", "1"
            ]
            
            logger.info(f"Running preprocessing without verification: {' '.join(cmd)}")
            print(f"Command: {' '.join(cmd)}\n")
            print("="*60)
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                universal_newlines=True
            )
            
            for line in process.stdout:
                print(line, end='')
                logger.info(line.strip())
            
            process.wait()
            
            if process.returncode == 0:
                print("\n" + "="*60)
                print("✓ Preprocessing Completed!")
                print("="*60)
                logger.info("Preprocessing completed without verification")
                return True
            else:
                print("\n" + "="*60)
                print("✗ Preprocessing Still Failed")
                print("="*60)
                print("\nThe dataset has fundamental issues.")
                print("Please verify your data manually.")
                logger.error(f"Preprocessing failed even without verification")
                return False
                
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            return False

    def train_model(self, args) -> bool:
        """Train nnUNet model with automatic preprocessing."""
        try:
            logger.info(f"Starting model training for Dataset{args.task_id:03d}")
            
            dataset_manager = DatasetManager(self.nnunet_raw)
            existing_datasets = dataset_manager.get_existing_datasets()
            
            if not existing_datasets:
                logger.error("No datasets found for training")
                print("\n" + "="*60)
                print("ERROR: No Datasets Available")
                print("="*60)
                return False
            
            dataset_info = None
            for ds in existing_datasets:
                if ds['id'] == args.task_id:
                    dataset_info = ds
                    break
            
            if not dataset_info:
                logger.error(f"Dataset {args.task_id} not found")
                return False
            
            dataset_name = f"Dataset{args.task_id:03d}_{dataset_info['name']}"
            dataset_path = Path(self.nnunet_raw) / dataset_name
            
            logger.info(f"Training dataset: {dataset_name}")
            logger.info(f"Dataset path: {dataset_path}")
            
            if not dataset_path.exists():
                logger.error(f"Dataset path not found: {dataset_path}")
                return False
            
            images_tr = dataset_path / "imagesTr"
            labels_tr = dataset_path / "labelsTr"
            images_ts = dataset_path / "imagesTs"
            
            if not images_tr.exists() or not labels_tr.exists():
                logger.error("Missing required folders")
                return False
            
            train_images = list(images_tr.glob("*.nii.gz"))
            train_labels = list(labels_tr.glob("*.nii.gz"))
            test_images = list(images_ts.glob("*.nii.gz")) if images_ts.exists() else []
            
            num_train_images = len(train_images)
            num_train_labels = len(train_labels)
            num_test_images = len(test_images)
            
            logger.info(f"Training files: {num_train_images} images, {num_train_labels} labels")
            logger.info(f"Test files: {num_test_images} images")
            
            if num_train_images == 0 or num_train_labels == 0:
                logger.error("No training data found")
                return False
            
            print("\n" + "="*60)
            print("Training Configuration")
            print("="*60)
            print(f"Dataset: {dataset_name}")
            print(f"ID: {args.task_id}")
            print(f"Device: {getattr(args, 'device', 'cuda')}")
            print(f"Fold: {getattr(args, 'fold', 0)}")
            print(f"\nDataset Statistics:")
            print(f"  Training images: {num_train_images}")
            print(f"  Training labels: {num_train_labels}")
            print(f"  Test images: {num_test_images}")
            
            print(f"\nSample files:")
            print("Images:")
            for img in train_images[:3]:
                print(f"  ✓ {img.name}")
            if num_train_images > 3:
                print(f"  ... and {num_train_images - 3} more")
            
            print("\nLabels:")
            for lbl in train_labels[:3]:
                print(f"  ✓ {lbl.name}")
            if num_train_labels > 3:
                print(f"  ... and {num_train_labels - 3} more")
            
            preprocessed_dirs = list(Path(self.nnunet_preprocessed).glob(f"Dataset{args.task_id:03d}_*"))
            plans_exists = False
            
            if preprocessed_dirs:
                for prep_dir in preprocessed_dirs:
                    plans_file = prep_dir / "nnUNetPlans.json"
                    if plans_file.exists():
                        plans_exists = True
                        logger.info(f"Found existing plans: {plans_file}")
                        break
            
            if not plans_exists:
                print("\n" + "="*60)
                print("⚠️  Preprocessing Required")
                print("="*60)
                print("\nThis is the first time training this dataset.")
                print("nnUNet needs to analyze and preprocess the data.")
                print("This may take 10-30 minutes depending on dataset size.")
                
                preprocess_confirm = input("\nRun preprocessing now? [Y/n]: ").strip().lower()
                if preprocess_confirm and preprocess_confirm != 'y':
                    print("Training cancelled.")
                    return False
                
                if not self._run_preprocessing(args.task_id):
                    logger.error("Preprocessing failed")
                    print("\nCannot proceed with training.")
                    return False
            else:
                logger.info("Preprocessing already completed, skipping...")
                print("\n✓ Dataset already preprocessed")
            
            print("\n" + "="*60)
            print("⚠️  Training may take several hours")
            print("="*60)
            
            if num_train_images < 30:
                print(f"\n⚠️  WARNING: Only {num_train_images} training samples")
                print("   nnUNet recommends 30+ samples for good results")
            
            confirm = input("\nStart training? [Y/n]: ").strip().lower()
            if confirm and confirm != 'y':
                print("Training cancelled")
                return False
            
            device = getattr(args, 'device', 'cuda')
            fold = getattr(args, 'fold', 0)
            
            cmd = [
                "nnUNetv2_train",
                str(args.task_id),
                "3d_fullres",
                str(fold),
                "-device", device
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            print(f"\nExecuting: {' '.join(cmd)}\n")
            print("="*60)
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                universal_newlines=True
            )
            
            try:
                for line in process.stdout:
                    print(line, end='')
                    logger.info(line.strip())
            except KeyboardInterrupt:
                print("\n\nTraining interrupted by user.")
                process.terminate()
                return False
            
            process.wait()
            
            if process.returncode == 0:
                logger.info("Training completed successfully")
                print("\n" + "="*60)
                print("✓ Training Completed Successfully!")
                print("="*60)
                return True
            else:
                logger.error(f"Training failed: {process.returncode}")
                print("\n" + "="*60)
                print("✗ Training Failed")
                print("="*60)
                return False
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def predict_model(self, args) -> bool:
        """Run predictions using trained model."""
        try:
            trainer = NNUNetTrainer(
                self.nnunet_raw,
                self.nnunet_preprocessed,
                self.nnunet_results
            )
            return trainer.predict(args)
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return False

    def visualize_results(self, args) -> bool:
        """Visualize predictions."""
        try:
            visualizer = Visualizer()
            return visualizer.visualize(args)
        except Exception as e:
            logger.error(f"Visualization error: {str(e)}")
            return False

    def download_synapse(self, args) -> bool:
        """Download data from Synapse."""
        try:
            downloader = SynapseDownloader()
            return downloader.download(args.synapse_id, self.downloads_dir)
        except Exception as e:
            logger.error(f"Synapse download error: {str(e)}")
            return False

    def upload_synapse(self, args) -> bool:
        """Upload results to Synapse."""
        try:
            uploader = SynapseUploader()
            return uploader.upload(args.project_id, args.results_path)
        except Exception as e:
            logger.error(f"Synapse upload error: {str(e)}")
            return False

    def download_zenodo(self, args) -> bool:
        """Download data from Zenodo."""
        try:
            downloader = ZenodoDownloader()
            return downloader.download(args.record_id, self.downloads_dir)
        except Exception as e:
            logger.error(f"Zenodo download error: {str(e)}")
            return False

    def organize_with_ai(self, args) -> bool:
        """Organize and convert datasets using AI."""
        try:
            ai_organizer = AIOrganizer(self.nnunet_raw)
            return ai_organizer.organize_zip_with_ai()
        except Exception as e:
            logger.error(f"AI organization error: {str(e)}")
            return False


