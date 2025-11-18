"""
AI-Powered Dataset Organization
Automatically organizes medical imaging datasets for nnUNet
"""

import os
import json
import shutil
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("ai_organizer")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not installed. AI features disabled.")

from src.dataset_manager import DatasetManager

class AIOrganizer:
    """AI-powered dataset organization with automatic train/test detection."""
    
    def __init__(self, nnunet_raw_dir: str):
        """Initialize AI Organizer."""
        self.nnunet_raw_dir = nnunet_raw_dir
        self.dataset_manager = DatasetManager(nnunet_raw_dir)
        
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and OPENAI_AVAILABLE:
                self.client = OpenAI(api_key=api_key)
                self.ai_available = True
                logger.info("OpenAI API initialized successfully")
            else:
                self.ai_available = False
                logger.info("OpenAI API key not found, manual mode active")
        except Exception as e:
            self.ai_available = False
            logger.info(f"OpenAI initialization failed: {e}, manual mode active")

    def organize_zip_with_ai(self) -> bool:
        """Organize datasets from zip files, automatically combining train/test sets."""
        try:
            raw_dir = Path(self.nnunet_raw_dir).parent.parent / "raw"
            
            if not raw_dir.exists():
                logger.error(f"Raw data directory not found: {raw_dir}")
                return False
            
            zip_files = list(raw_dir.glob("**/*.zip"))
            
            if not zip_files:
                logger.error(f"No zip files found in {raw_dir}")
                print(f"\nNo datasets found in: {raw_dir}")
                print("\nPlease place your dataset zip files in:")
                print(f"  {raw_dir}")
                return False
            
            print(f"\nFound {len(zip_files)} zip file(s):")
            for i, zip_file in enumerate(zip_files, 1):
                print(f"  {i}. {zip_file.name}")
            
            # Group files by base name (detect train/test pairs)
            grouped_files = self._group_train_test_files(zip_files)
            
            if grouped_files:
                print(f"\nDetected {len(grouped_files)} dataset(s) with train/test pairs:")
                for group_name, files in grouped_files.items():
                    print(f"\n  Dataset: {group_name}")
                    for file_type, file_path in files.items():
                        print(f"    - {file_type}: {file_path.name}")
                
                # Process all grouped datasets
                all_success = True
                for group_name, files in grouped_files.items():
                    print(f"\n{'='*60}")
                    print(f"Processing: {group_name}")
                    print(f"{'='*60}")
                    
                    success = self._process_train_test_pair(group_name, files)
                    if not success:
                        all_success = False
                        logger.error(f"Failed to process {group_name}")
                    
                return all_success
            else:
                # No train/test pairs detected, process individually
                print("\nNo train/test pairs detected. Processing files individually...")
                
                if len(zip_files) == 1:
                    selected_zip = zip_files[0]
                    print(f"\nAuto-selected: {selected_zip.name}")
                else:
                    try:
                        choice = int(input(f"\nSelect dataset [1-{len(zip_files)}]: ").strip() or "1")
                        selected_zip = zip_files[choice - 1]
                    except (ValueError, IndexError):
                        selected_zip = zip_files[0]
                        print(f"Using first dataset: {selected_zip.name}")
                
                return self._process_single_zip(selected_zip)
            
        except Exception as e:
            logger.error(f"Error organizing datasets: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _group_train_test_files(self, zip_files: List[Path]) -> Dict[str, Dict[str, Path]]:
        """Group train and test zip files by base name."""
        groups = {}
        
        for zip_file in zip_files:
            name_lower = zip_file.stem.lower()
            
            # Remove common keywords to find base name
            base_name = name_lower
            for keyword in ['train', 'test', 'training', 'testing', 'val', 'validation', 'set', '_', '-']:
                base_name = base_name.replace(keyword, '')
            
            base_name = base_name.strip('_- ')
            
            if not base_name:
                base_name = zip_file.stem
            
            # Determine file type
            if any(keyword in name_lower for keyword in ['train', 'training']):
                file_type = 'train'
            elif any(keyword in name_lower for keyword in ['test', 'testing', 'val', 'validation']):
                file_type = 'test'
            else:
                file_type = 'unknown'
            
            # Group files
            if base_name not in groups:
                groups[base_name] = {}
            
            groups[base_name][file_type] = zip_file
        
        # Filter out groups that don't have both train and test
        complete_groups = {}
        for base_name, files in groups.items():
            if 'train' in files and 'test' in files:
                complete_groups[base_name] = files
            elif 'train' in files:
                logger.info(f"Found train set without test: {base_name}")
            elif 'test' in files:
                logger.info(f"Found test set without train: {base_name}")
        
        return complete_groups

    def _process_train_test_pair(self, group_name: str, files: Dict[str, Path]) -> bool:
        """Process a pair of train and test zip files."""
        try:
            train_zip = files.get('train')
            test_zip = files.get('test')
            
            if not train_zip or not test_zip:
                logger.error(f"Missing train or test file for {group_name}")
                return False
            
            # Create temporary extraction directories
            temp_train_dir = train_zip.parent / f"temp_train_{group_name}"
            temp_test_dir = test_zip.parent / f"temp_test_{group_name}"
            
            temp_train_dir.mkdir(exist_ok=True)
            temp_test_dir.mkdir(exist_ok=True)
            
            print(f"\nExtracting train set: {train_zip.name}")
            with zipfile.ZipFile(train_zip, 'r') as zip_ref:
                zip_ref.extractall(temp_train_dir)
            
            print(f"Extracting test set: {test_zip.name}")
            with zipfile.ZipFile(test_zip, 'r') as zip_ref:
                zip_ref.extractall(temp_test_dir)
            
            # Analyze structures
            train_structure = self._analyze_structure(temp_train_dir)
            test_structure = self._analyze_structure(temp_test_dir)
            
            print(f"\nTrain set: {len(train_structure['images'])} images, {len(train_structure['labels'])} labels")
            print(f"Test set: {len(test_structure['images'])} images, {len(test_structure['labels'])} labels")
            
            if not train_structure['images'] or not train_structure['labels']:
                logger.error("Invalid train set structure")
                shutil.rmtree(temp_train_dir)
                shutil.rmtree(temp_test_dir)
                return False
            
            # Get dataset name
            dataset_name = input(f"\nEnter dataset name [default: {group_name}]: ").strip()
            if not dataset_name:
                dataset_name = group_name.replace('_', '').replace('-', '')
            
            dataset_id = self.dataset_manager.get_next_dataset_id()
            
            # Create combined dataset
            success = self._create_combined_dataset(
                train_structure,
                test_structure,
                dataset_id,
                dataset_name
            )
            
            # Cleanup
            shutil.rmtree(temp_train_dir)
            shutil.rmtree(temp_test_dir)
            
            if success:
                print(f"\n{'='*60}")
                print("Dataset Created Successfully!")
                print(f"{'='*60}")
                print(f"Name: Dataset{dataset_id:03d}_{dataset_name}")
                print(f"ID: {dataset_id}")
                print(f"Training samples: {len(train_structure['images'])}")
                print(f"Test samples: {len(test_structure['images'])}")
                print(f"Location: {self.nnunet_raw_dir}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing train/test pair: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _process_single_zip(self, zip_path: Path) -> bool:
        """Process a single zip file with manual train/test split."""
        try:
            logger.info(f"Processing zip file: {zip_path.name}")
            
            temp_dir = zip_path.parent / f"temp_{zip_path.stem}"
            temp_dir.mkdir(exist_ok=True)
            
            print(f"\nExtracting: {zip_path.name}")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            structure = self._analyze_structure(temp_dir)
            
            if not structure['images'] or not structure['labels']:
                logger.error("Invalid dataset structure")
                shutil.rmtree(temp_dir)
                return False
            
            print(f"\nDataset: {len(structure['images'])} images, {len(structure['labels'])} labels")
            
            dataset_name = input("\nEnter dataset name: ").strip()
            if not dataset_name:
                dataset_name = zip_path.stem.replace('_', '').replace('-', '')
            
            dataset_id = self.dataset_manager.get_next_dataset_id()
            
            # Ask for split ratio
            split_input = input("\nTrain/Test split (e.g., 0.8) [default: 0.8]: ").strip()
            try:
                train_ratio = float(split_input) if split_input else 0.8
                if not 0 < train_ratio < 1:
                    train_ratio = 0.8
            except ValueError:
                train_ratio = 0.8
            
            success = self._create_dataset_with_split(
                structure,
                dataset_id,
                dataset_name,
                train_ratio
            )
            
            shutil.rmtree(temp_dir)
            
            if success:
                print(f"\n{'='*60}")
                print("Dataset Created Successfully!")
                print(f"{'='*60}")
                print(f"Name: Dataset{dataset_id:03d}_{dataset_name}")
                print(f"ID: {dataset_id}")
                print(f"Train: {int(len(structure['images']) * train_ratio)} samples")
                print(f"Test: {int(len(structure['images']) * (1 - train_ratio))} samples")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing single zip: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _analyze_structure(self, root_dir: Path) -> Dict:
        """Analyze dataset structure."""
        structure = {
            'images': [],
            'labels': [],
            'format': None
        }
        
        image_extensions = ['.nii', '.nii.gz', '.dcm', '.png', '.jpg', '.jpeg']
        
        for file_path in root_dir.rglob("*"):
            if not file_path.is_file():
                continue
            
            if any(file_path.name.endswith(ext) for ext in image_extensions):
                path_lower = str(file_path).lower()
                
                if any(keyword in path_lower for keyword in ['label', 'mask', 'seg', 'gt', 'ground']):
                    structure['labels'].append(file_path)
                else:
                    structure['images'].append(file_path)
        
        structure['images'].sort()
        structure['labels'].sort()
        
        if structure['images']:
            structure['format'] = structure['images'][0].suffix
        
        return structure

    def _create_combined_dataset(
        self,
        train_structure: Dict,
        test_structure: Dict,
        dataset_id: int,
        dataset_name: str
    ) -> bool:
        """Create nnUNet dataset from separate train and test sets."""
        try:
            dataset_folder = Path(self.nnunet_raw_dir) / f"Dataset{dataset_id:03d}_{dataset_name}"
            
            images_tr_dir = dataset_folder / "imagesTr"
            labels_tr_dir = dataset_folder / "labelsTr"
            images_ts_dir = dataset_folder / "imagesTs"
            labels_ts_dir = dataset_folder / "labelsTs"
            
            for dir_path in [images_tr_dir, labels_tr_dir, images_ts_dir, labels_ts_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Copy training files
            print("\nCopying training files...")
            for idx, (img_path, lbl_path) in enumerate(zip(train_structure['images'], train_structure['labels'])):
                case_id = f"case_{idx:04d}"
                
                img_dest = images_tr_dir / f"{case_id}_0000.nii.gz"
                lbl_dest = labels_tr_dir / f"{case_id}.nii.gz"
                
                shutil.copy2(img_path, img_dest)
                shutil.copy2(lbl_path, lbl_dest)
                
                if (idx + 1) % 10 == 0:
                    print(f"  Processed {idx + 1}/{len(train_structure['images'])} training samples")
            
            # Copy test files
            print("\nCopying test files...")
            for idx, (img_path, lbl_path) in enumerate(zip(test_structure['images'], test_structure['labels'])):
                case_id = f"case_{idx:04d}"
                
                img_dest = images_ts_dir / f"{case_id}_0000.nii.gz"
                lbl_dest = labels_ts_dir / f"{case_id}.nii.gz"
                
                shutil.copy2(img_path, img_dest)
                shutil.copy2(lbl_path, lbl_dest)
                
                if (idx + 1) % 10 == 0:
                    print(f"  Processed {idx + 1}/{len(test_structure['images'])} test samples")
            
            # Create dataset.json
            self._create_dataset_json(
                dataset_folder,
                dataset_name,
                len(train_structure['images']),
                len(test_structure['images'])
            )
            
            logger.info(f"Combined dataset created: {dataset_folder}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating combined dataset: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _create_dataset_with_split(
        self,
        structure: Dict,
        dataset_id: int,
        dataset_name: str,
        train_ratio: float
    ) -> bool:
        """Create dataset with train/test split from single source."""
        try:
            dataset_folder = Path(self.nnunet_raw_dir) / f"Dataset{dataset_id:03d}_{dataset_name}"
            
            images_tr_dir = dataset_folder / "imagesTr"
            labels_tr_dir = dataset_folder / "labelsTr"
            images_ts_dir = dataset_folder / "imagesTs"
            labels_ts_dir = dataset_folder / "labelsTs"
            
            for dir_path in [images_tr_dir, labels_tr_dir, images_ts_dir, labels_ts_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            total_samples = min(len(structure['images']), len(structure['labels']))
            train_count = int(total_samples * train_ratio)
            
            print(f"\nSplitting dataset: {train_count} train, {total_samples - train_count} test")
            
            for idx, (img_path, lbl_path) in enumerate(zip(structure['images'][:total_samples], structure['labels'][:total_samples])):
                case_id = f"case_{idx:04d}"
                
                if idx < train_count:
                    img_dest = images_tr_dir / f"{case_id}_0000.nii.gz"
                    lbl_dest = labels_tr_dir / f"{case_id}.nii.gz"
                else:
                    img_dest = images_ts_dir / f"{case_id}_0000.nii.gz"
                    lbl_dest = labels_ts_dir / f"{case_id}.nii.gz"
                
                shutil.copy2(img_path, img_dest)
                shutil.copy2(lbl_path, lbl_dest)
            
            self._create_dataset_json(dataset_folder, dataset_name, train_count, total_samples - train_count)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating split dataset: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _create_dataset_json(
        self,
        dataset_folder: Path,
        dataset_name: str,
        num_training: int,
        num_test: int
    ):
        """Create dataset.json file."""
        dataset_json = {
            "channel_names": {
                "0": "CT"
            },
            "labels": {
                "background": 0,
                "organ": 1
            },
            "numTraining": num_training,
            "numTest": num_test,
            "file_ending": ".nii.gz",
            "name": dataset_name,
            "description": f"Dataset: {dataset_name}",
            "reference": "Auto-generated",
            "license": "Unknown",
            "release": "1.0"
        }
        
        json_path = dataset_folder / "dataset.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_json, f, indent=2)
        
        logger.info(f"Created dataset.json: {json_path}")

    def organize_zip_with_ai_json_nested(self) -> bool:
        """Alternative organization method (for option 9)."""
        return self.organize_zip_with_ai()