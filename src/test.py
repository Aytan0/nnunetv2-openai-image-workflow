import os
import subprocess

os.putenv("nnUNet_raw", "C:\\Users\\aytan\\OneDrive\\Masaüstü\\A\\synapse-nnunet-app\\data\\nnUNet\\nnUNet_raw")
os.putenv("nnUNet_preprocessed", "C:\\Users\\aytan\\OneDrive\\Masaüstü\\A\\synapse-nnunet-app\\data\\nnUNet\\nnUNet_preprocessed")
os.putenv("nnUNet_results", "C:\\Users\\aytan\\OneDrive\\Masaüstü\\A\\synapse-nnunet-app\\data\\nnUNet\\nnUNet_results")

# Now run the nnUNetv2 command
subprocess.run(["nnUNetv2_plan_and_preprocess", "-d", "501", "--verify_dataset_integrity"])