import pytest
import os
import shutil
from pathlib import Path
from nnunet_trainer import NNUNetTrainer

# Define a test task ID and directory
TEST_TASK_ID = 999
TEST_TASK_NAME = "TestTask"
TEST_DATA_DIR = Path("./test_data")
NNUNET_RAW_DIR = TEST_DATA_DIR / "nnUNet_raw"
NNUNET_TASK_DIR = NNUNET_RAW_DIR / f"Dataset{TEST_TASK_ID:03d}_{TEST_TASK_NAME}"
NNUNET_IMAGES_TR_DIR = NNUNET_TASK_DIR / "imagesTr"
NNUNET_LABELS_TR_DIR = NNUNET_TASK_DIR / "labelsTr"

# Create test data directory if it doesn't exist
TEST_DATA_DIR.mkdir(exist_ok=True)

# Create nnUNet_raw directory if it doesn't exist
NNUNET_RAW_DIR.mkdir(exist_ok=True)

# Create a dummy dataset
NNUNET_TASK_DIR.mkdir(exist_ok=True)
NNUNET_IMAGES_TR_DIR.mkdir(parents=True, exist_ok=True)
NNUNET_LABELS_TR_DIR.mkdir(parents=True, exist_ok=True)

# Create dummy image and label files
with open(NNUNET_IMAGES_TR_DIR / "test_image_0000.nii.gz", "w") as f:
    f.write("dummy image data")
with open(NNUNET_LABELS_TR_DIR / "test_label.nii.gz", "w") as f:
    f.write("dummy label data")

@pytest.fixture
def trainer():
    # Set environment variables for testing
    os.environ['nnUNet_raw'] = str(NNUNET_RAW_DIR)
    os.environ['nnUNet_preprocessed'] = str(TEST_DATA_DIR / "nnUNet_preprocessed")
    os.environ['nnUNet_results'] = str(TEST_DATA_DIR / "nnUNet_results")
    print(f"nnUNet_raw: {os.environ['nnUNet_raw']}")
    print(f"nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
    print(f"nnUNet_results: {os.environ['nnUNet_results']}")
    return NNUNetTrainer()

def test_preprocess_dataset_success(trainer):
    result = trainer.preprocess_dataset(TEST_TASK_ID)
    assert result is True
    # Add assertions to check if the directories and files are created correctly
    assert (TEST_DATA_DIR / "nnUNet_preprocessed").exists()
    assert (TEST_DATA_DIR / "nnUNet_results").exists()

def test_preprocess_dataset_failure(trainer):
    # Remove the nnUNet_raw directory to simulate a failure
    shutil.rmtree(NNUNET_RAW_DIR)
    result = trainer.preprocess_dataset(TEST_TASK_ID)
    assert result is False
    # Restore the nnUNet_raw directory
    NNUNET_RAW_DIR.mkdir(exist_ok=True)

# Clean up the test data directory after all tests are done
def teardown_module():
    shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)