import os
import sys
import argparse
import logging
import traceback
import shutil
import time
import locale
from pathlib import Path
from dotenv import load_dotenv
import synapseclient
from src.workflow import WorkflowManager
import io

def setup_logging():
    """Set up logging with safe encoding for all platforms."""
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # FIX: Console handler için encoding belirt
    file_handler = logging.FileHandler(
        os.path.join(logs_dir, 'synapse_nnunet.log'), 
        encoding='utf-8',
        errors='replace'
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # FIX: Console handler için UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Windows için encoding fix
    if os.name == 'nt':
        import io
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, 
            encoding='utf-8', 
            errors='replace'
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, 
            encoding='utf-8', 
            errors='replace'
        )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()  # Clear existing handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger = logging.getLogger('main')
    logger.info("Logging initialized successfully")
    return logger

logger = setup_logging()

def show_terminal_helper():
    """Display a terminal-based helper for command selection."""
    print("\n" + "="*80)
    print("nnUNetV2 Terminal Helper - Enhanced Version".center(80))
    print("="*80)
    print("NEW FEATURES:")
    print("   Automatic Dataset ID System (starting from 001)")
    print("   Smart Dataset Selection (auto-scan)")
    print("   80% Fewer Questions (smart defaults)")
    print("   Enhanced Error Messages and Solutions")
    print("="*80)
    
    print("\nPlease select the operation you want to perform:")
    print("1. Download data from Synapse (with ID)")
    print("2. Download data from Zenodo (with Records ID)")
    print("3. Data conversion (Automatic Dataset Selection)")
    print("4. Model training")
    print("5. Make predictions")
    print("6. Visualize results")
    print("7. Run full workflow")
    print("8. Clear nnUNet data")
    print("9. AI-Powered Data Organization (Automatic)")
    print("10. List existing datasets")
    print("11. Exit")
    
    choice = input("\nYour choice (1-11): ")
    
    args = argparse.Namespace()
    args.task_name = "dataset"
    args.task_id = None
    args.device = "cuda"
    args.fold = 0
    
    if choice in ["2"]:
        device_choice = input("\nWhich device do you want to use? [1: CUDA (GPU), 2: CPU, default: CUDA]: ")
        if device_choice == "2":
            args.device = "cpu"
        
        fold_choice = input("Which fold do you want to train? [0-4, default: 0]: ")
        try:
            args.fold = int(fold_choice) if fold_choice.strip() else 0
        except ValueError:
            args.fold = 0
            print("Default fold 0 will be used.")
    
    if choice in ["4", "5", "6", "7"]:
        custom_name = input("Do you want to use a custom dataset name? [leave empty = automatic]: ").strip()
        if custom_name:
            args.task_name = custom_name
    
    return choice, args

def handle_synapse_download(workflow_manager, args):
    args.synapse_project_id = input("Synapse Project ID: ")
    if workflow_manager.download_synapse_dataset(args):
        logger.info("Synapse data download completed successfully.")
    else:
        logger.error("Synapse data download failed.")

def handle_zenodo_download(workflow_manager, args):
    args.zenodo_record_id = input("Zenodo Records ID: ")
    if workflow_manager.download_zenodo_dataset(args):
        logger.info("Zenodo data download completed successfully.")
    else:
        logger.error("Zenodo data download failed.")

def handle_conversion(workflow_manager, args):
    print("\nAutomatic Dataset Conversion Starting...")
    print("This feature automatically finds potential datasets and lets you choose")
    
    if workflow_manager.convert_datasets_auto():
        logger.info("Data conversion completed successfully.")
        print("\nRecommendation: Check existing datasets with option 10")
    else:
        logger.error("Data conversion failed.")
        print("\nSolutions:")
        print("   1. Check that raw data is in the correct directories:")
        print("      - data/raw/")
        print("      - data/zip/") 
        print("      - downloads/")
        print("   2. Ensure data is in NIfTI format (.nii.gz)")
        print("   3. Check folder structure (imagesTr, labelsTr)")
        print("   4. Check log files: logs/")
        print("   5. Ensure sufficient disk space")
                
def handle_training(workflow_manager, args, helper_args):
    args.task_name = helper_args.task_name
    from src.dataset_manager import DatasetManager
    dataset_manager = DatasetManager(workflow_manager.nnunet_raw)
    existing_datasets = dataset_manager.get_existing_datasets()
    
    if not existing_datasets:
        logger.error("No dataset found for training. Do data conversion first (option 3).")
        return
    
    if len(existing_datasets) == 1:
        selected_dataset = existing_datasets[0]
        args.task_id = selected_dataset['id']
        print(f"Auto-selected dataset: {selected_dataset['name']} (ID: {args.task_id})")
    else:
        print("\nExisting datasets:")
        for i, ds in enumerate(existing_datasets):
            print(f"  {i+1}. {ds['name']} (ID: {ds['id']})")
        
        try:
            selection = int(input("Which dataset do you want to train? [1]: ") or 1) - 1
            args.task_id = existing_datasets[selection]['id']
        except (ValueError, IndexError):
            args.task_id = existing_datasets[0]['id']
            print(f"Using default dataset: ID {args.task_id}")
    
    args.device = helper_args.device
    args.fold = helper_args.fold
    
    try:
        if hasattr(workflow_manager, 'train_model') and workflow_manager.train_model(args):
            logger.info(f"Model training completed successfully (Task: {args.task_id}, Fold: {args.fold})")
        else:
            logger.error(f"Model training failed (Task: {args.task_id}, Fold: {args.fold})")
    except AttributeError:
        logger.error("train_model method not found. Please check workflow.py.")

def handle_prediction(workflow_manager, args, helper_args):
    args.task_name = helper_args.task_name
    from src.dataset_manager import DatasetManager
    dataset_manager = DatasetManager(workflow_manager.nnunet_raw)
    existing_datasets = dataset_manager.get_existing_datasets()
    
    if not existing_datasets:
        logger.error("No dataset found for prediction. Do data conversion first (option 3).")
        return
        
    args.task_id = existing_datasets[-1]['id']
    print(f"Auto-selected dataset: {existing_datasets[-1]['name']} (ID: {args.task_id})")
    args.device = helper_args.device
    
    if workflow_manager.predict_model(args):
        logger.info("Prediction completed successfully.")
    else:
        logger.error("Prediction failed.")

def handle_visualization(workflow_manager, args, helper_args):
    args.task_name = helper_args.task_name
    from src.dataset_manager import DatasetManager
    dataset_manager = DatasetManager(workflow_manager.nnunet_raw)
    existing_datasets = dataset_manager.get_existing_datasets()
    
    if not existing_datasets:
        logger.error("No dataset found for visualization. Do data conversion first (option 3).")
        return
        
    args.task_id = existing_datasets[-1]['id']
    print(f"Auto-selected dataset: {existing_datasets[-1]['name']} (ID: {args.task_id})")
    
    if workflow_manager.visualize_results(args):
        logger.info("Results visualized successfully.")
    else:
        logger.error("Results could not be visualized.")

def handle_full_workflow(workflow_manager, args, helper_args):
    args.task_name = helper_args.task_name
    args.device = helper_args.device

    print("\nStarting Full Workflow...")
    download_choice = input("Which source do you want to download data from? [1: Synapse, 2: Chaos]: ")
    
    if download_choice == "1":
        args.synapse_project_id = input("Synapse Project ID: ")
        download_success = workflow_manager.download_synapse_dataset(args)
    elif download_choice == "2":
        args.zenodo_record_id = input("Zenodo Records ID: ")
        download_success = workflow_manager.download_zenodo_dataset(args)
    else:
        logger.error("Invalid download source selection.")
        return

    if not download_success:
        logger.error("Data download failed.")
        return

    logger.info("Starting automatic dataset conversion...")
    if not workflow_manager.convert_datasets_auto():
        logger.error("Dataset conversion failed.")
        return
    
    logger.info("Download and conversion completed successfully.")
    
    from src.dataset_manager import DatasetManager
    dataset_manager = DatasetManager(workflow_manager.nnunet_raw)
    existing_datasets = dataset_manager.get_existing_datasets()
    
    if not existing_datasets:
        logger.error("Converted dataset not found.")
        return
    
    latest_dataset = existing_datasets[-1]
    args.task_id = latest_dataset['id']
    args.task_name = latest_dataset['name']
    
    print(f"Auto-selected dataset: {args.task_name} (ID: {args.task_id})")

    logger.info("Starting nnUNet model training...")
    try:
        if not (hasattr(workflow_manager, 'train_model') and workflow_manager.train_model(args)):
            logger.error("nnUNet model training failed.")
            return
            
        logger.info("nnUNet model training completed successfully.")

        logger.info("Visualizing results...")
        if workflow_manager.visualize_results(args):
            logger.info("Results visualized and exported successfully.")
        else:
            logger.error("Results could not be visualized.")
    except AttributeError:
        logger.error("train_model method not found.")

def handle_clear_data(workflow_manager, args):
    workflow_manager.clear_nnunet_data()

def handle_ai_organization(workflow_manager, args):
    print("\nStarting AI-Powered Data Organization...")
    print("This feature automatically analyzes and organizes your data")
    
    if workflow_manager.organize_data_with_ai():
        logger.info("AI-powered data organization completed successfully.")
        print("\nNext Step: Start model training with option 4")
    else:
        logger.error("AI-powered data organization failed.")
        print("\nSolutions:")
        print("   1. Check your OpenAI API key (.env file)")
        print("   2. Check your internet connection")
        print("   3. Manual mode: Use option 3 (Automatic Dataset Selection)")
        print("   4. Check that raw data is in the correct directories")
        print("   5. Supported formats: NIfTI (.nii.gz), DICOM")
                
def handle_list_datasets(workflow_manager, args):
    print("\nListing Existing Datasets...")
    workflow_manager.list_existing_datasets()
    print("\nTips:")
    print("   - At least one dataset is required for training")
    print("   - Datasets are stored in data/nnUNet/nnUNet_raw/")  
    print("   - Use option 3 to add new datasets")

def handle_exit(workflow_manager, args):
    print("Exiting program...")
    sys.exit(0)

# Dictionary mapping choices to handler functions
MENU_HANDLERS = {
    "1": handle_synapse_download,
    "2": handle_zenodo_download,
    "3": handle_conversion,
    "4": handle_training,
    "5": handle_prediction,
    "6": handle_visualization,
    "7": handle_full_workflow,
    "8": handle_clear_data,
    "9": handle_ai_organization,
    "10": handle_list_datasets,
    "11": handle_exit,
}

def main(argv=None):
    """Main function to parse arguments and run the workflow."""
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    load_dotenv()
    
    workflow_manager = WorkflowManager()

    parser = argparse.ArgumentParser(description="Synapse-nnUNet CLI")
    subparsers = parser.add_subparsers(dest="command")

    parser_setup = subparsers.add_parser('setup', help="Set up the environment")
    parser_download = subparsers.add_parser('download', help="Download datasets from Synapse.org")
    parser_download.add_argument('--task-name', type=str, help="Task name (e.g., LiverVesselSeg)")
    parser_download.add_argument('--task-id', type=int, help="Task ID (e.g., 501)")
    parser_download.add_argument('--force-refresh', action='store_true', help='Force refresh of dataset download')
    parser_download.add_argument('--zenodo-record-id', type=str, help='Zenodo Records ID (e.g., 3431873)')

    parser_convert = subparsers.add_parser('convert', help="Convert datasets to nnUNet format")
    parser_convert.add_argument("--input-folder", type=str, help="Input folder containing raw dataset")
    parser_convert.add_argument("--task-name", type=str, help="Task name (e.g., LiverVesselSeg)")
    parser_convert.add_argument('--task-id', type=int, help="Task ID (e.g., 501)")
    parser_convert.add_argument('--force-refresh', action='store_true', help="Force refresh of dataset conversion")

    parser_train = subparsers.add_parser('train', help="Train the nnUNet model")
    parser_train.add_argument('--task-id', type=int, help="Task ID (e.g., 501)")
    parser_train.add_argument('--device', type=str, help="Device to use for training (cuda or cpu)")
    parser_train.add_argument('--task-name', type=str, help="Task name (e.g., LiverVesselSeg)")
    parser_train.add_argument('--fold', type=int, default=0, help="Fold to train (0-4)")

    parser_predict = subparsers.add_parser('predict', help="Run prediction with the nnUNet model")
    parser_predict.add_argument('--task-id', type=int, help="Task ID (e.g., 501)")
    parser_predict.add_argument('--device', type=str, help="Device to use for prediction (cuda or cpu)")
    parser_predict.add_argument('--task-name', type=str, help="Task name (e.g., LiverVesselSeg)")

    parser_visualize = subparsers.add_parser('visualize', help="Visualize the results of the nnUNet model")
    parser_visualize.add_argument('--task-id', type=int, help="Task ID (e.g., 501)")

    parser_clear = subparsers.add_parser('clear', help="Clear nnUNet data directories")

    parser_upload = subparsers.add_parser('upload', help="Upload results to Synapse.org")
    parser_upload.add_argument('--synapse-project-id', type=str, help="Synapse Project ID to upload results to")

    args = parser.parse_args(argv)

    if not args.command:
        choice, helper_args = show_terminal_helper()
        
        # Use dictionary dispatch instead of elif chain
        handler = MENU_HANDLERS.get(choice)
        if handler:
            # Handle functions that need helper_args
            if choice in ["4", "5", "6", "7"]:
                handler(workflow_manager, args, helper_args)
            else:
                handler(workflow_manager, args)
        else:
            print("Invalid choice.")
            
    elif args.command == 'setup':
        logger.info("Environment setup is handled automatically by WorkflowManager.")
        
    elif args.command == 'download':
        if workflow_manager.download_datasets(args):
            logger.info(f"Automatically running convert command: --task-name {args.task_name} --task-id {args.task_id}")
            if workflow_manager.convert_datasets(args, input_folder=workflow_manager.raw_data_dir):
                logger.info("Download and convert commands completed successfully.")

                logger.info("Starting nnUNet model training...")
                if workflow_manager.train_model(args):
                    logger.info("nnUNet model training completed successfully.")

                    logger.info("Visualizing and exporting results...")
                    if workflow_manager.visualize_results(args):
                        logger.info("Results visualized and exported successfully.")
                    else:
                        logger.error("Results could not be visualized and exported.")
                else:
                    logger.error("nnUNet model training failed.")
            else:
                logger.error("Convert command failed.")
                
    elif args.command == 'convert':
        workflow_manager.convert_datasets(args)
        
    elif args.command == 'train':
        workflow_manager.train_model(args)
        
    elif args.command == 'predict':
        workflow_manager.predict_model(args)
        
    elif args.command == 'visualize':
        workflow_manager.visualize_results(args)
        
    elif args.command == 'clear':
        workflow_manager.clear_nnunet_data()
        
    elif args.command == 'upload':
        workflow_manager.upload_results_to_synapse(args)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

