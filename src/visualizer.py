"""
Results Visualizer and Exporter

This module handles the visualization and export of nnUNet training and prediction results.
It provides functionality to generate summary statistics, visualize segmentations,
and export results to various formats.
"""

import os
import json
import shutil
import logging
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'results_visualizer.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('results_visualizer')

class Visualizer:
    """Class to visualize and export nnUNet results."""
    
    def __init__(self, nnunet_data_dir=None, export_dir=None):
        """
        Initialize the ResultsVisualizer.
        
        Args:
            nnunet_data_dir (str, optional): Directory for nnUNet data.
                                            Defaults to '../data/nnUNet'.
            export_dir (str, optional): Directory to export results.
                                       Defaults to '../data/exports'.
        """
        # Load environment variables
        load_dotenv()
        
        # Set directories
        base_dir = os.path.dirname(os.path.dirname(__file__))
        
        if nnunet_data_dir is None:
            self.nnunet_data_dir = os.path.join(base_dir, 'data', 'nnUNet')
        else:
            self.nnunet_data_dir = nnunet_data_dir
            
        if export_dir is None:
            self.export_dir = os.path.join(base_dir, 'data', 'exports')
        else:
            self.export_dir = export_dir
        
        # Set nnUNet environment variables if not already set
        if not os.getenv('nnUNet_raw'):
            os.environ['nnUNet_raw'] = os.path.join(self.nnunet_data_dir, 'nnUNet_raw')
        if not os.getenv('nnUNet_preprocessed'):
            os.environ['nnUNet_preprocessed'] = os.path.join(self.nnunet_data_dir, 'nnUNet_preprocessed')
        if not os.getenv('nnUNet_results'):
            os.environ['nnUNet_results'] = os.path.join(self.nnunet_data_dir, 'nnUNet_results')
        
        # Create export directory if it doesn't exist
        os.makedirs(self.export_dir, exist_ok=True)
        
        logger.info(f"nnUNet data directory: {self.nnunet_data_dir}")
        logger.info(f"Export directory: {self.export_dir}")
    
    def generate_training_summary(self, task_id, export=True):
        """
        Generate a summary of training results.
        
        Args:
            task_id (int): ID of the task.
            export (bool, optional): Whether to export the summary. Defaults to True.
            
        Returns:
            dict: Summary of training results.
        """
        try:
            logger.info(f"Generating training summary for Task {task_id}...")
            
            # Find all training results for this task
            results_dir = os.environ['nnUNet_results']
            task_results = []
            
            # Look for results in the results directory
            for config_dir in os.listdir(results_dir):
                if not os.path.isdir(os.path.join(results_dir, config_dir)):
                    continue
                    
                task_dir_pattern = f"Task{task_id}_"
                task_dirs = [d for d in os.listdir(os.path.join(results_dir, config_dir)) if d.startswith(task_dir_pattern)]
                
                for task_dir in task_dirs:
                    task_path = os.path.join(results_dir, config_dir, task_dir)
                    
                    # Look for fold directories
                    fold_dirs = [d for d in os.listdir(task_path) if d.startswith("fold_")]
                    
                    for fold_dir in fold_dirs:
                        fold_path = os.path.join(task_path, fold_dir)
                        
                        # Check for validation metrics
                        validation_file = os.path.join(fold_path, "validation", "summary.json")
                        if os.path.exists(validation_file):
                            with open(validation_file, 'r') as f:
                                validation_data = json.load(f)
                                
                                # Add to results
                                task_results.append({
                                    "config": config_dir,
                                    "task": task_dir,
                                    "fold": fold_dir,
                                    "metrics": validation_data
                                })
            
            if not task_results:
                logger.warning(f"WARNING: No training results found for Task {task_id}")
                return None
            
            # Compile summary
            summary = {
                "task_id": task_id,
                "num_configurations": len(set(r["config"] for r in task_results)),
                "num_folds": len(set(r["fold"] for r in task_results)),
                "configurations": list(set(r["config"] for r in task_results)),
                "results": task_results,
                "best_result": max(task_results, key=lambda x: x["metrics"].get("mean", {}).get("Dice", 0))
            }
            
            # Export summary if requested
            if export:
                export_path = os.path.join(self.export_dir, f"task_{task_id}_training_summary.json")
                with open(export_path, 'w') as f:
                    json.dump(summary, f, indent=4)
                logger.info(f"Exported training summary to {export_path}")
            
            logger.info(f"Generated training summary for Task {task_id}")
            return summary
            
        except Exception as e:
            logger.error(f"ERROR: Failed to generate training summary: {str(e)}")
            return None
    
    def visualize_segmentations(self, prediction_dir, output_dir=None, num_slices=5):
        """
        Visualize segmentation results.
        
        Args:
            prediction_dir (str): Directory containing prediction results.
            output_dir (str, optional): Directory to save visualizations.
                                       Defaults to a subdirectory of export_dir.
            num_slices (int, optional): Number of slices to visualize per volume.
                                       Defaults to 5.
            
        Returns:
            bool: True if visualization successful, False otherwise.
        """
        try:
            logger.info(f"Visualizing segmentations from {prediction_dir}...")
            
            # Set output directory
            if output_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join(self.export_dir, f"visualizations_{timestamp}")
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Find all segmentation files
            seg_files = [f for f in os.listdir(prediction_dir) if f.endswith('.nii.gz')]
            
            if not seg_files:
                logger.warning(f"WARNING: No segmentation files found in {prediction_dir}")
                return False
            
            logger.info(f"Found {len(seg_files)} segmentation files")
            
            # Process each segmentation file
            for seg_file in seg_files:
                try:
                    seg_path = os.path.join(prediction_dir, seg_file)
                    
                    # Load segmentation
                    seg_nii = nib.load(seg_path)
                    seg_data = seg_nii.get_fdata()
                    
                    # Get dimensions
                    depth = seg_data.shape[2]
                    
                    # Select slices to visualize
                    slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
                    
                    # Create figure
                    fig, axes = plt.subplots(1, num_slices, figsize=(4*num_slices, 4))
                    if num_slices == 1:
                        axes = [axes]
                    
                    # Plot each slice
                    for i, slice_idx in enumerate(slice_indices):
                        axes[i].imshow(seg_data[:, :, slice_idx], cmap='viridis')
                        axes[i].set_title(f"Slice {slice_idx}")
                        axes[i].axis('off')
                    
                    # Save figure
                    output_file = os.path.join(output_dir, f"{os.path.splitext(seg_file)[0]}_visualization.png")
                    plt.tight_layout()
                    plt.savefig(output_file, dpi=150)
                    plt.close()
                    
                    logger.info(f"Saved visualization for {seg_file} to {output_file}")
                    
                except Exception as e:
                    logger.error(f"ERROR: Failed to visualize {seg_file}: {str(e)}")
            
            logger.info(f"Visualization completed. Results saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"ERROR: Failed to visualize segmentations: {str(e)}")
            return False
    
    def export_results(self, task_id, prediction_dir=None, export_format="nifti"):
        """
        Export prediction results to a specified format.
        
        Args:
            task_id (int): ID of the task.
            prediction_dir (str, optional): Directory containing prediction results.
                                          If None, will look in default location.
            export_format (str, optional): Format to export results.
                                          Options: "nifti", "png".
                                          Defaults to "nifti".
            
        Returns:
            str: Path to exported results or None if export failed.
        """
        try:
            # If prediction_dir is not provided, use default location
            if prediction_dir is None:
                prediction_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "predictions")
            
            logger.info(f"Exporting results for Task {task_id} from {prediction_dir}...")
            
            # Create export directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = os.path.join(self.export_dir, f"task_{task_id}_results_{timestamp}")
            os.makedirs(export_path, exist_ok=True)
            
            # Check if prediction directory exists
            if not os.path.exists(prediction_dir):
                logger.error(f"ERROR: Prediction directory {prediction_dir} does not exist")
                return None
            
            # Find all files in prediction directory
            pred_files = [f for f in os.listdir(prediction_dir) if f.endswith('.nii.gz')]
            
            if not pred_files:
                logger.warning(f"WARNING: No prediction files found in {prediction_dir}")
                return None
            
            logger.info(f"Found {len(pred_files)} prediction files")
            
            # Export based on format
            if export_format.lower() == "nifti":
                # Simply copy the files
                for pred_file in pred_files:
                    src_path = os.path.join(prediction_dir, pred_file)
                    dst_path = os.path.join(export_path, pred_file)
                    shutil.copy2(src_path, dst_path)
                    logger.info(f"Copied {pred_file} to {dst_path}")
                
            elif export_format.lower() == "png":
                # Export as PNG slices
                for pred_file in pred_files:
                    try:
                        pred_path = os.path.join(prediction_dir, pred_file)
                        
                        # Load prediction
                        pred_nii = nib.load(pred_path)
                        pred_data = pred_nii.get_fdata()
                        
                        # Create directory for this file
                        file_dir = os.path.join(export_path, os.path.splitext(pred_file)[0])
                        os.makedirs(file_dir, exist_ok=True)
                        
                        # Export each slice
                        for slice_idx in range(pred_data.shape[2]):
                            slice_data = pred_data[:, :, slice_idx]
                            
                            # Create figure
                            plt.figure(figsize=(8, 8))
                            plt.imshow(slice_data, cmap='viridis')
                            plt.axis('off')
                            
                            # Save figure
                            slice_file = os.path.join(file_dir, f"slice_{slice_idx:04d}.png")
                            plt.savefig(slice_file, dpi=150, bbox_inches='tight')
                            plt.close()
                        
                        logger.info(f"Exported {pred_file} as PNG slices to {file_dir}")
                        
                    except Exception as e:
                        logger.error(f"ERROR: Failed to export {pred_file} as PNG: {str(e)}")
                
            else:
                logger.error(f"ERROR: Unsupported export format: {export_format}")
                return None
            
            # Create a summary file
            summary = {
                "task_id": task_id,
                "export_format": export_format,
                "num_files": len(pred_files),
                "files": pred_files,
                "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            summary_path = os.path.join(export_path, "export_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=4)
            
            logger.info(f"Export completed. Results saved to {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"ERROR: Failed to export results: {str(e)}")
            return None

def export_results(task_id):
    """
    Export results for a specific task.
    
    Args:
        task_id (int): ID of the task.
        
    Returns:
        bool: True if export successful, False otherwise.
    """
    try:
        # Create visualizer
        visualizer = ResultsVisualizer()
        
        # Export results
        prediction_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "predictions")
        export_path = visualizer.export_results(task_id, prediction_dir)
        
        if export_path:
            # Visualize segmentations
            visualizer.visualize_segmentations(prediction_dir)
            
            # Generate training summary
            visualizer.generate_training_summary(task_id)
            
            return True
        else:
            return False
            
    except Exception as e:
        logger.error(f"ERROR: Failed to export results: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    task_id = 501  # Example task ID
    export_results(task_id)
