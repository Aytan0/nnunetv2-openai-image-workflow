"""
Dataset Converter for nnUNet v2

This module handles the conversion of downloaded datasets from Synapse.org
into the format required by nnUNet v2 for training.
"""

import os
import re
import shutil
import zipfile
import logging
import nibabel as nib
import numpy as np
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import json
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
# Ensure logs directory exists before configuring logging
logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'dataset_converter.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('dataset_converter')

class DatasetConverter:
    """Class to handle dataset conversion to nnUNet format."""
    
    def __init__(self):
        """Initialize DatasetConverter."""
        self.raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
        self.nnunet_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'nnUNet')
        self.nnunet_raw = os.path.join(self.nnunet_data_dir, 'nnUNet_raw')
        self.nnunet_preprocessed = os.path.join(self.nnunet_data_dir, 'nnUNet_preprocessed')
        self.nnunet_results = os.path.join(self.nnunet_data_dir, 'nnUNet_results')
        
        # Ensure nnUNet directories exist
        os.makedirs(self.nnunet_raw, exist_ok=True)
        os.makedirs(self.nnunet_preprocessed, exist_ok=True)
        os.makedirs(self.nnunet_results, exist_ok=True)
        
        # Ensure logs directory exists
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        logger.info(f"Raw data directory: {self.raw_data_dir}")
        logger.info(f"nnUNet data directory: {self.nnunet_data_dir}")
        logger.info(f"nnUNet_raw: {self.nnunet_raw}")
        logger.info(f"nnUNet_preprocessed: {self.nnunet_preprocessed}")
        logger.info(f"nnUNet_results: {self.nnunet_results}")
    
    def list_datasets(self):
        """
        List all available datasets in the raw data directory.
        
        Returns:
            list: List of dataset folder names in the raw data directory.
        """
        try:
            if not os.path.exists(self.raw_data_dir):
                logger.warning(f"Raw data directory does not exist: {self.raw_data_dir}")
                return []
            
            datasets = []
            for item in os.listdir(self.raw_data_dir):
                item_path = os.path.join(self.raw_data_dir, item)
                if os.path.isdir(item_path):
                    datasets.append(item)
                    logger.info(f"Found dataset: {item}")
            
            logger.info(f"Total datasets found: {len(datasets)}")
            return sorted(datasets)
            
        except Exception as e:
            logger.error(f"Error listing datasets: {str(e)}")
            return []
    
    def get_next_dataset_id(self):
        """
        Get the next available dataset ID by checking existing datasets in nnUNet_raw.
        
        Returns:
            int: Next available dataset ID.
        """
        try:
            if not os.path.exists(self.nnunet_raw):
                return 1
            
            existing_ids = []
            for item in os.listdir(self.nnunet_raw):
                if item.startswith("Dataset") and os.path.isdir(os.path.join(self.nnunet_raw, item)):
                    # Extract ID from Dataset###_Name format
                    match = re.match(r"Dataset(\d{3})_", item)
                    if match:
                        existing_ids.append(int(match.group(1)))
            
            if not existing_ids:
                return 1
            
            return max(existing_ids) + 1
            
        except Exception as e:
            logger.error(f"Error getting next dataset ID: {str(e)}")
            return 1
    
    def select_dataset(self, dataset_name):
        """
        Select a dataset from the raw data directory.
        
        Args:
            dataset_name (str): Name of the dataset folder to select.
            
        Returns:
            str: Full path to the selected dataset, or None if not found.
        """
        try:
            dataset_path = os.path.join(self.raw_data_dir, dataset_name)
            
            if not os.path.exists(dataset_path):
                logger.error(f"Dataset not found: {dataset_name}")
                return None
            
            if not os.path.isdir(dataset_path):
                logger.error(f"Dataset path is not a directory: {dataset_path}")
                return None
            
            logger.info(f"Selected dataset: {dataset_name} at {dataset_path}")
            return dataset_path
            
        except Exception as e:
            logger.error(f"Error selecting dataset {dataset_name}: {str(e)}")
            return None
    
    def convert_dataset(self, dataset_name, custom_task_name=None, custom_task_id=None):
        """
        Convert a selected dataset to nnUNet format.
        
        Args:
            dataset_name (str): Name of the dataset folder in raw data directory.
            custom_task_name (str, optional): Custom task name. If None, uses dataset_name.
            custom_task_id (int, optional): Custom task ID. If None, uses next available ID.
            
        Returns:
            dict: Result dictionary with success status and details.
        """
        try:
            # Select the dataset
            dataset_path = self.select_dataset(dataset_name)
            if not dataset_path:
                return {
                    "success": False,
                    "error": f"Dataset '{dataset_name}' not found or invalid."
                }
            
            # Determine task name and ID
            task_name = custom_task_name if custom_task_name else dataset_name
            task_id = custom_task_id if custom_task_id else self.get_next_dataset_id()
            
            logger.info(f"Converting dataset '{dataset_name}' to nnUNet format...")
            logger.info(f"Task Name: {task_name}")
            logger.info(f"Task ID: {task_id}")
            
            # Organize the dataset
            success = self.organize_dataset(dataset_path, task_name, task_id)
            
            if success:
                result_dataset_name = f"Dataset{task_id:03d}_{task_name}"
                result_path = os.path.join(self.nnunet_raw, result_dataset_name)
                
                return {
                    "success": True,
                    "dataset_name": result_dataset_name,
                    "task_id": task_id,
                    "task_name": task_name,
                    "output_path": result_path,
                    "message": f"Dataset successfully converted to {result_dataset_name}"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to organize dataset. Check logs for details."
                }
                
        except Exception as e:
            logger.error(f"Error converting dataset {dataset_name}: {str(e)}")
            return {
                "success": False,
                "error": f"Conversion failed: {str(e)}"
            }
    
    def _extract_zip_files(self, input_path: str) -> str:
        """
        ZIP dosyalarını geçici dizine çıkar.
        
        Args:
            input_path (str): ZIP dosyası veya ZIP dosyaları içeren dizin
            
        Returns:
            str: Çıkarılmış dosyaların bulunduğu dizin yolu
        """
        # Geçici dizin oluştur
        temp_dir = tempfile.mkdtemp(prefix="nnunet_extract_")
        logger.info(f"ZIP dosyaları geçici dizine çıkarılıyor: {temp_dir}")
        
        zip_files = []
        
        # Eğer input_path direkt bir zip dosyası ise
        if input_path.endswith('.zip') and os.path.isfile(input_path):
            zip_files.append(input_path)
        
        # Eğer input_path bir dizin ise, içindeki zip dosyalarını bul
        elif os.path.isdir(input_path):
            zip_files = list(Path(input_path).rglob("*.zip"))
            zip_files = [str(f) for f in zip_files]
        
        if not zip_files:
            logger.warning(f"Hiç ZIP dosyası bulunamadı: {input_path}")
            return input_path
        
        logger.info(f"Bulunan ZIP dosyaları: {len(zip_files)}")
        
        # Her zip dosyasını çıkar
        for zip_file in zip_files:
            try:
                logger.info(f"ZIP dosyası çıkarılıyor: {os.path.basename(zip_file)}")
                
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    # ZIP içeriğini kontrol et
                    file_list = zip_ref.namelist()
                    logger.info(f"ZIP içinde {len(file_list)} dosya bulundu")
                    
                    # Çıkar
                    zip_ref.extractall(temp_dir)
                    
                    # İlk birkaç dosyayı logla
                    for i, filename in enumerate(file_list[:5]):
                        logger.info(f"  Çıkarıldı: {filename}")
                    if len(file_list) > 5:
                        logger.info(f"  ... ve {len(file_list) - 5} dosya daha")
                
                logger.info(f"✅ ZIP başarıyla çıkarıldı: {os.path.basename(zip_file)}")
                
            except zipfile.BadZipFile:
                logger.error(f"❌ Bozuk ZIP dosyası: {zip_file}")
                continue
            except Exception as e:
                logger.error(f"❌ ZIP çıkarma hatası {zip_file}: {e}")
                continue
        
        # Çıkarılan içeriği kontrol et
        extracted_items = os.listdir(temp_dir)
        logger.info(f"Çıkarılan öğeler: {extracted_items}")
        
        return temp_dir
    
    def _cleanup_temp_dir(self, temp_dir: str):
        """Geçici dizini temizle."""
        try:
            if os.path.exists(temp_dir) and temp_dir.startswith(tempfile.gettempdir()):
                shutil.rmtree(temp_dir)
                logger.info(f"Geçici dizin temizlendi: {temp_dir}")
        except Exception as e:
            logger.warning(f"Geçici dizin temizleme hatası: {e}")
    
    def organize_dataset(self, input_folder, task_name, task_id):
        """
        Organize the extracted dataset into the nnUNet format.
        Otomatik ZIP dosyası desteği dahil.
        
        Args:
            input_folder (str): Kaynak dizin veya ZIP dosyası yolu
            task_name (str): Name of the task.
            task_id (int): ID of the task.
            
        Returns:
            bool: True if organization successful, False otherwise.
        """
        temp_dir = None
        try:
            logger.info(f"Organizing dataset with task_id: {task_id} and task_name: {task_name}...")
            logger.info(f"Kaynak: {input_folder}")
            
            # ZIP dosyası kontrolü ve çıkarma
            working_dir = input_folder
            
            # ZIP dosyası var mı kontrol et
            has_zip = False
            if input_folder.endswith('.zip') and os.path.isfile(input_folder):
                has_zip = True
                logger.info("🗜️ Direkt ZIP dosyası tespit edildi")
            elif os.path.isdir(input_folder):
                zip_files = list(Path(input_folder).rglob("*.zip"))
                if zip_files:
                    has_zip = True
                    logger.info(f"🗜️ Dizin içinde {len(zip_files)} ZIP dosyası tespit edildi")
            
            # ZIP dosyası varsa çıkar
            if has_zip:
                logger.info("📦 ZIP dosyaları çıkarılıyor...")
                temp_dir = self._extract_zip_files(input_folder)
                working_dir = temp_dir
                logger.info(f"✅ ZIP çıkarma tamamlandı, çalışma dizini: {working_dir}")
            else:
                logger.info("📁 Normal dizin, ZIP çıkarma gerekli değil")
            
            # Define paths
            dataset_name = f"Dataset{task_id:03d}_{task_name}"
            task_dir = os.path.join(self.nnunet_raw, dataset_name)
            imagesTr_dir = os.path.join(task_dir, "imagesTr")
            labelsTr_dir = os.path.join(task_dir, "labelsTr")
            imagesTs_dir = os.path.join(task_dir, "imagesTs")
            labelsTs_dir = os.path.join(task_dir, "labelsTs")
            
            # Check if the directory already exists
            if os.path.exists(task_dir):
                logger.warning(f"WARNING: Directory {task_dir} already exists. Deleting...")
                shutil.rmtree(task_dir)
            
            # Create directories
            os.makedirs(imagesTr_dir, exist_ok=True)
            os.makedirs(labelsTr_dir, exist_ok=True)
            os.makedirs(imagesTs_dir, exist_ok=True)
            os.makedirs(labelsTs_dir, exist_ok=True)
            
            # Dynamically find training and test data directories
            train_dirs = []
            test_dirs = []
            for root, dirs, _ in os.walk(working_dir):
                for d in dirs:
                    full_path = os.path.join(root, d)
                    if "train" in d.lower():
                        train_dirs.append(full_path)
                    elif "test" in d.lower():
                        test_dirs.append(full_path)

            # Process training data
            if train_dirs:
                for train_dir in train_dirs:
                    logger.info(f"Found training directory: {train_dir}")
                    self._process_data_directory(train_dir, imagesTr_dir, labelsTr_dir, is_training=True)
            else:
                logger.warning("WARNING: No training data directory found.")

            # Process test data
            if test_dirs:
                for test_dir in test_dirs:
                    logger.info(f"Found test directory: {test_dir}")
                    self._process_data_directory(test_dir, imagesTs_dir, labelsTs_dir, is_training=False)
            else:
                logger.warning("WARNING: No test data directory found.")
            
            # Create dataset.json
            self._create_dataset_json(task_dir, task_name, task_id)
            
            # Verify dataset.json was created
            dataset_json_path = os.path.join(task_dir, "dataset.json")
            if not os.path.exists(dataset_json_path):
                logger.error(f"ERROR: dataset.json was not created at {dataset_json_path}")
                return False
                
            logger.info(f"Dataset organization completed for {dataset_name}")
            
            # Clean up temporary directory if created
            if temp_dir:
                self._cleanup_temp_dir(temp_dir)
            
            return True
        
        except Exception as e:
            logger.error(f"ERROR: Failed to organize dataset: {str(e)}")
            # Clean up temporary directory if created
            if temp_dir:
                self._cleanup_temp_dir(temp_dir)
            return False
    
    def _process_data_directory(self, source_dir, images_dir, labels_dir, is_training=True):
        """
        Process data directory and copy files to nnUNet format.
        
        Args:
            source_dir (str): Source directory with extracted data.
            images_dir (str): Target directory for images.
            labels_dir (str): Target directory for labels.
            is_training (bool): Whether this is training data.
        """
        try:
            # Find all NIfTI files in the source directory
            nifti_files = list(Path(source_dir).rglob("*.nii.gz")) + list(Path(source_dir).rglob("*.nii"))

            for file_path in nifti_files:
                file = os.path.basename(file_path)
                # Determine if this is an image or label file
                is_label = False
                # Check for common label indicators in filename or parent directory name
                if 'gt' in file.lower() or 'mask' in file.lower() or 'seg' in file.lower():
                    is_label = True
                # Also check if the file is in a directory named 'labels' or 'segmentations'
                if 'labels' in str(file_path.parent).lower() or 'segmentations' in str(file_path.parent).lower():
                    is_label = True

                # Generate nnUNet-compatible filename
                case_id = self._extract_case_id(file)
                if not case_id:
                    logger.warning(f"WARNING: Could not extract case ID from {file}, using filename as case ID")
                    case_id = os.path.splitext(file)[0]
                    if case_id.endswith(".nii"):
                        case_id = os.path.splitext(case_id)[0]

                if is_label:
                    # Label files: case_id.nii.gz
                    target_filename = f"{case_id}.nii.gz"
                    target_path = os.path.join(labels_dir, target_filename)
                else:
                    # Image files: case_id_0000.nii.gz (assuming single modality for now, will improve later)
                    target_filename = f"{case_id}_0000.nii.gz"
                    target_path = os.path.join(images_dir, target_filename)

                # Copy and convert file if necessary
                self._copy_and_convert_file(file_path, target_path)               
            logger.info(f"Processed {'training' if is_training else 'test'} data directory: {source_dir}")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to process data directory {source_dir}: {str(e)}")
    
    def _extract_case_id(self, filename):
        """
        Extract case ID from filename using a more robust set of patterns.
        
        Args:
            filename (str): Original filename.
            
        Returns:
            str: Extracted case ID or None if not found.
        """
        # Define a list of regex patterns to try
        patterns = [
            r"^([a-zA-Z0-9]+)_(\d{4})\.nii\.gz$",  # e.g., case_0000.nii.gz (for images)
            r"^([a-zA-Z0-9]+)\.nii\.gz$",  # e.g., case.nii.gz (for labels)
            r"^([a-zA-Z0-9]+)_(\d+)_(\d{4})\.nii\.gz$", # e.g., patient_123_0000.nii.gz
            r"^([a-zA-Z0-9]+)_(\d+)\.nii\.gz$", # e.g., patient_123.nii.gz
            r"^([a-zA-Z0-9]+)-(\d+)\.nii\.gz$", # e.g., patient-123.nii.gz
            r"^([a-zA-Z0-9]+)_(\d+)\.nii$", # e.g., case_0000.nii
            r"^([a-zA-Z0-9]+)\.nii$", # e.g., case.nii
            r"^(\d+)_([a-zA-Z0-9]+)_(\d{4})\.nii\.gz$", # e.g., 001_case_0000.nii.gz
            r"^(\d+)_([a-zA-Z0-9]+)\.nii\.gz$", # e.g., 001_case.nii.gz
            r"^(\d+)_([a-zA-Z0-9]+)\.nii$", # e.g., 001_case.nii
            r"^([a-zA-Z0-9]+)", # General case for names without specific patterns
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                # Return the first group that is not a modality or a number if it's a name
                for g in match.groups():
                    if not re.match(r"^\d{4}$", g) and not re.match(r"^\d+$", g):
                        return g
                return match.group(1) # Fallback to first group if no non-modality/number group found
        
        # If no pattern matches, use the filename without extension as case ID
        base_name = os.path.basename(filename)
        name_without_ext = os.path.splitext(base_name)[0]
        if name_without_ext.endswith(".nii"):
            name_without_ext = os.path.splitext(name_without_ext)[0]
            
        return name_without_ext
    
    def _copy_and_convert_file(self, source_path, target_path):
        """
        Copy and convert file to nnUNet format if necessary.
        
        Args:
            source_path (str): Path to source file.
            target_path (str): Path to target file.
        """
        try:
            # Load the NIfTI file
            img = nib.load(source_path)
            
            # Save as .nii.gz (even if source is .nii)
            nib.save(img, target_path)
            
            logger.info(f"Copied and converted: {source_path} -> {target_path}")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to copy and convert {source_path}: {str(e)}")
    
    def _create_dataset_json(self, task_dir, task_name, task_id):
        """
        Create dataset.json file required by nnUNet.
        
        Args:
            task_dir (str): Task directory.
            task_name (str): Name of the task.
            task_id (int): ID of the task.
        """
        try:
            logger.info(f"Creating dataset.json for task {task_id}_{task_name}...")
            
            # Check if imagesTr directory exists and has files
            imagesTr_dir = os.path.join(task_dir, "imagesTr")
            if not os.path.exists(imagesTr_dir):
                logger.error(f"ERROR: imagesTr directory does not exist: {imagesTr_dir}")
                logger.error("ÇÖZÜM: Ham veri dizininde geçerli NIfTI dosyaları olduğundan emin olun.")
                return
                
            # Count training cases
            training_cases = [f for f in os.listdir(imagesTr_dir) if f.endswith('.nii.gz')]
            num_training = len(training_cases)
            
            if num_training == 0:
                logger.error(f"ERROR: No training images found in {imagesTr_dir}")
                logger.error("ÇÖZÜM: Ham veri dizininde geçerli NIfTI dosyaları olduğundan emin olun.")
                return
            
            logger.info(f"Found {num_training} training cases")
            
            # Count test cases
            imagesTs_dir = os.path.join(task_dir, "imagesTs")
            test_cases = []
            if os.path.exists(imagesTs_dir):
                test_cases = [f for f in os.listdir(imagesTs_dir) if f.endswith('.nii.gz')]
            num_test = len(test_cases)
            
            logger.info(f"Found {num_test} test cases")
            
            # Create dataset.json structure
            dataset_json = {
                "channel_names": {
                    "0": "CT"  # Default to CT, can be modified based on actual data
                },
                "labels": {
                    "background": 0,
                    "target": 1  # Default binary segmentation, can be modified
                },
                "numTraining": num_training,
                "numTest": num_test,
                "file_ending": ".nii.gz",
                "dataset_name": task_name,
                "task_id": task_id
            }
            
            # Save dataset.json
            dataset_json_path = os.path.join(task_dir, "dataset.json")
            with open(dataset_json_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_json, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ dataset.json created successfully at {dataset_json_path}")
            logger.info(f"Dataset contains {num_training} training and {num_test} test cases")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to create dataset.json: {str(e)}")


# CLI interface for testing
if __name__ == "__main__":
    converter = DatasetConverter()
    
    # List available datasets
    print("Available datasets:")
    datasets = converter.list_datasets()
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset}")
    
    if not datasets:
        print("No datasets found in raw data directory.")
        exit(1)
    
    # Get user selection
    try:
        choice = int(input(f"Select dataset (1-{len(datasets)}): ")) - 1
        if choice < 0 or choice >= len(datasets):
            print("Invalid selection.")
            exit(1)
        
        selected_dataset = datasets[choice]
        print(f"Selected dataset: {selected_dataset}")
        
        # Ask for custom naming
        use_custom = input("Do you want to use custom task name and ID? (y/n): ").lower().strip()
        
        custom_task_name = None
        custom_task_id = None
        
        if use_custom == 'y':
            custom_task_name = input("Enter custom task name: ").strip()
            custom_task_id = int(input("Enter custom task ID: "))
        
        # Convert dataset
        result = converter.convert_dataset(selected_dataset, custom_task_name, custom_task_id)
        
        if result["success"]:
            print(f"✅ {result['message']}")
            print(f"Output path: {result['output_path']}")
        else:
            print(f"❌ Conversion failed: {result['error']}")
            
    except (ValueError, KeyboardInterrupt):
        print("Operation cancelled.")
        exit(1)

