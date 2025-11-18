import os
import sys
import argparse
import subprocess
import platform
import json
import logging
from pathlib import Path
import requests
import zipfile
import io
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("setup_nnunet_conda")

class CondaSetup:
    """Class to handle conda environment setup for nnUNetV2."""
    
    def __init__(self, env_name="nnunetv2", cuda_version="11.7", python_version="3.12"):
        """
        Initialize the CondaSetup.
        
        Args:
            env_name (str): Name of the conda environment to create or use.
            cuda_version (str): CUDA version to use for PyTorch.
            python_version (str): Python version to use in the conda environment.
        """
        self.env_name = env_name
        self.cuda_version = cuda_version
        self.python_version = python_version
        self.is_windows = platform.system() == "Windows"
        self.conda_executable = "conda"
        self.pip_executable = "pip"
        
        # Set paths
        self.script_dir = Path(__file__).parent.absolute()
        self.extras_dir = self.script_dir.parent / "extras"
        self.graphviz_dir = self.extras_dir / "Graphviz"
        
        logger.info(f"Initializing conda setup for nnUNetV2")
        logger.info(f"Environment name: {self.env_name}")
        logger.info(f"CUDA version: {self.cuda_version}")
        logger.info(f"Python version: {self.python_version}")
        logger.info(f"Operating system: {platform.system()}")
    def _run_command(self, cmd, description, check=True, capture_output=True):
        logger.info(f"Running {description}: {' '.join(map(str, cmd))}")
        try:
            result = subprocess.run(
                cmd,
                stderr=subprocess.PIPE,
                text=True,
                check=check
            )
            if result.stdout:
                logger.info(f"{description} stdout:\n{result.stdout.strip()}")
            if result.stderr:
                logger.error(f"{description} stderr:\n{result.stderr.strip()}")
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"ERROR: {description} failed with exit code {e.returncode}.\nStdout: {e.stdout}\nStderr: {e.stderr}")
            return None
        except FileNotFoundError:
            logger.error(f"ERROR: Command not found. Make sure {cmd[0]} is in your PATH.")
            return None
        except Exception as e:
            logger.error(f"ERROR: An unexpected error occurred during {description}: {str(e)}")
            return None
    
    def check_conda_installed(self):
        """
        Check if conda is installed and accessible.
        
        Returns:
            bool: True if conda is installed, False otherwise.
        """
        try:
            logger.info("Checking if conda is installed...")
            result = subprocess.run(
                [self.conda_executable, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                logger.info(f"Conda is installed: {result.stdout.strip()}")
                return True
            else:
                logger.error("Conda is not installed or not in PATH")
                logger.error(f"Error: {result.stderr.strip()}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking conda installation: {str(e)}")
            return False
    
    def check_env_exists(self):
        """
        Check if the specified conda environment exists.
        
        Returns:
            bool: True if environment exists, False otherwise.
        """
        try:
            logger.info(f"Checking if conda environment \'{self.env_name}\' exists...")
            result = subprocess.run(
                [self.conda_executable, "env", "list", "--json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                env_list = json.loads(result.stdout)
                env_names = [os.path.basename(env) for env in env_list.get("envs", [])]
                
                if self.env_name in env_names:
                    logger.info(f"Conda environment \'{self.env_name}\' exists")
                    return True
                else:
                    logger.info(f"Conda environment \'{self.env_name}\' does not exist")
                    return False
            else:
                logger.error(f"Error listing conda environments: {result.stderr.strip()}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking conda environment: {str(e)}")
            return False
    
    def create_conda_env(self):
        """
        Create a new conda environment with the specified Python version.
        
        Returns:
            bool: True if environment creation successful, False otherwise.
        """
        try:
            if self.check_env_exists():
                logger.info(f"Using existing conda environment \'{self.env_name}\'")
                return True
                
            logger.info(f"Creating conda environment \'{self.env_name}\' with Python {self.python_version}...")
            result = subprocess.run(
                [self.conda_executable, "create", "-n", self.env_name, f"python={self.python_version}", "-y"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully created conda environment \'{self.env_name}\'")
                return True
            else:
                logger.error(f"Failed to create conda environment: {result.stderr.strip()}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating conda environment: {str(e)}")
            return False
    
    def install_pytorch(self):
        """
        Install PyTorch with CUDA support in the conda environment.
        
        Returns:
            bool: True if installation successful, False otherwise.
        """
        try:
            logger.info(f"Installing PyTorch with CUDA {self.cuda_version} support...")
            
            # Determine PyTorch installation command based on CUDA version
            if self.cuda_version == "11.7":
                pytorch_install_cmd = "pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia"
            elif self.cuda_version == "11.8":
                pytorch_install_cmd = "pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia"
            elif self.cuda_version == "12.1":
                pytorch_install_cmd = "pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia"
            else:
                pytorch_install_cmd = "pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia"
            
            # Install PyTorch
            result = subprocess.run(
                [self.conda_executable, "install", "-n", self.env_name, "-y"] + pytorch_install_cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                logger.info("Successfully installed PyTorch with CUDA support")
                return True
            else:
                logger.error(f"Failed to install PyTorch: {result.stderr.strip()}")
                return False
                
        except Exception as e:
            logger.error(f"Error installing PyTorch: {str(e)}")
            return False
    
    def install_nnunetv2(self):
        """
        Install nnUNetV2 and its dependencies in the conda environment.
        
        Returns:
            bool: True if installation successful, False otherwise.
        """
        try:
            logger.info("Installing nnUNetV2 and its dependencies...")
            
            # Install nnUNetV2 using pip in the conda environment
            result = subprocess.run(
                [self.conda_executable, "run", "-n", self.env_name, "pip", "install", "nnunetv2"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                logger.info("Successfully installed nnUNetV2")
                return True
            else:
                logger.error(f"Failed to install nnUNetV2: {result.stderr.strip()}")
                return False
                
        except Exception as e:
            logger.error(f"Error installing nnUNetV2: {str(e)}")
            return False
    
    def install_hiddenlayer(self):
        """
        Install hiddenlayer for visualization in the conda environment.
        
        Returns:
            bool: True if installation successful, False otherwise.
        """
        try:
            logger.info("Installing hiddenlayer for visualization...")
            
            # Install hiddenlayer using pip in the conda environment
            result = subprocess.run(
                [self.conda_executable, "run", "-n", self.env_name, "pip", "install", "hiddenlayer"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                logger.info("Successfully installed hiddenlayer")
                return True
            else:
                logger.error(f"Failed to install hiddenlayer: {result.stderr.strip()}")
                
                # Try installing from GitHub if PyPI fails
                logger.info("Trying to install hiddenlayer from GitHub...")
                result = subprocess.run(
                    [self.conda_executable, "run", "-n", self.env_name, "pip", "install", "git+https://github.com/waleedka/hiddenlayer.git"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0:
                    logger.info("Successfully installed hiddenlayer from GitHub")
                    return True
                else:
                    logger.error(f"Failed to install hiddenlayer from GitHub: {result.stderr.strip()}")
                    return False
                
        except Exception as e:
            logger.error(f"Error installing hiddenlayer: {str(e)}")
            return False
    
    def install_additional_dependencies(self):
        """
        Install additional dependencies required for the project.
        
        Returns:
            bool: True if installation successful, False otherwise.
        """
        try:
            logger.info("Installing additional dependencies...")
            
            # List of additional dependencies
            dependencies = [
                "nibabel",
                "matplotlib",
                "tqdm",
                "scikit-learn",
                "scikit-image",
                "SimpleITK",
                "pandas",
                "dotenv",
                "synapseclient"
            ]
            
            # Install dependencies using pip in the conda environment
            result = subprocess.run(
                [self.conda_executable, "run", "-n", self.env_name, "pip", "install"] + dependencies,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                logger.info("Successfully installed additional dependencies")
                return True
            else:
                logger.error(f"Failed to install additional dependencies: {result.stderr.strip()}")
                return False
                
        except Exception as e:
            logger.error(f"Error installing additional dependencies: {str(e)}")
            return False

    def download_and_setup_graphviz(self):
        logger.info("Downloading and setting up Graphviz...")
        if self.graphviz_dir.exists():
            logger.info("Graphviz directory already exists. Skipping download.")
            return True

        # Determine the correct Graphviz download URL based on the operating system
        if self.is_windows:
            # Example for Windows (you might need to find the exact latest stable release URL)
            # This is a placeholder URL, you'd need to find the actual download link for a portable version
            download_url = "https://gitlab.com/graphviz/graphviz/-/releases/permalink/windows_nsis-3.0.1_64-bit/downloads/graphviz-3.0.1-win64.zip" # This URL is likely outdated or incorrect, find a reliable one
            logger.warning(f"Automatic download for Windows Graphviz is complex due to installer vs portable versions. Attempting a generic download, but manual download from {download_url} might be necessary.")
            try:
                response = requests.get(download_url, stream=True)
                response.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                    # Extract to a temporary directory first to find the actual Graphviz folder name
                    temp_extract_dir = self.extras_dir / "temp_graphviz_extract"
                    zf.extractall(temp_extract_dir)
                    
                    # Find the actual Graphviz folder (e.g., graphviz-3.0.1-win64)
                    extracted_folder = None
                    for item in os.listdir(temp_extract_dir):
                        if "graphviz" in item.lower() and os.path.isdir(temp_extract_dir / item):
                            extracted_folder = temp_extract_dir / item
                            break
                    
                    if extracted_folder:
                        shutil.move(extracted_folder, self.graphviz_dir)
                        logger.info(f"Successfully downloaded and extracted Graphviz to {self.graphviz_dir}")
                        shutil.rmtree(temp_extract_dir)
                        return True
                    else:
                        logger.error("Could not find Graphviz folder inside the downloaded zip.")
                        shutil.rmtree(temp_extract_dir)
                        return False
            except Exception as e:
                logger.error(f"Failed to download or extract Graphviz for Windows: {e}")
                return False
        else:
            logger.warning("Graphviz is typically installed via package managers on Linux/macOS. Please install it manually: `sudo apt-get install graphviz` (Ubuntu) or `brew install graphviz` (macOS).")
            return False

    def setup_graphviz_path(self):
        logger.info("Setting up Graphviz PATH...")
        graphviz_bin_path = self.graphviz_dir / "bin"
        if self.is_windows:
            if graphviz_bin_path.exists():
                os.environ["PATH"] = str(graphviz_bin_path) + os.pathsep + os.environ["PATH"]
                logger.info(f"Added {graphviz_bin_path} to PATH for current session.")
                return True
            else:
                logger.error(f"Graphviz bin directory not found at {graphviz_bin_path}. Please ensure Graphviz is extracted correctly.")
                return False
        else:
            logger.warning("Graphviz setup is primarily for Windows. On Linux/macOS, please ensure Graphviz is installed via your system\'s package manager (e.g., `sudo apt-get install graphviz` or `brew install graphviz`).")
            return False

    def validate_installation(self):
        """
        Validate the installation by importing key packages.
        
        Returns:
            bool: True if validation successful, False otherwise.
        """
        try:
            logger.info("Validating installation...")
            
            # Create a validation script
            validation_script = """
import torch
import nibabel
import matplotlib
import SimpleITK
import pandas

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))
print("Validation successful!")
"""
            
            # Write validation script to a temporary file
            temp_script_path = os.path.join(self.script_dir, "temp_validation.py")
            with open(temp_script_path, "w") as f:
                f.write(validation_script)
            
            # Run the validation script in the conda environment
            result = subprocess.run(
                [self.conda_executable, "run", "-n", self.env_name, "python", temp_script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            # Remove the temporary script
            try:
                os.remove(temp_script_path)
            except:
                pass
            
            if result.returncode == 0:
                logger.info("Validation output:")
                for line in result.stdout.strip().split("\n"):
                    logger.info(f"  {line}")
                logger.info("Installation validation successful!")
                return True
            else:
                logger.error(f"Validation failed: {result.stderr.strip()}")
                return False
                
        except Exception as e:
            logger.error(f"Error validating installation: {str(e)}")
            return False
    
    def check_missing_dependencies(self):
        """
        Check for any missing dependencies.
        
        Returns:
            list: List of missing dependencies, empty if none.
        """
        try:
            logger.info("Checking for missing dependencies...")
            
            # List of essential dependencies to check
            essential_deps = [
                "torch",
                "nibabel",
                "matplotlib",
                "SimpleITK",
                "pandas",
                "tqdm",
                "sklearn",
                "skimage",
                "nnunetv2"
            ]
            
            missing_deps = []
            
            # Create a check script
            check_script = "import sys\n"
            for dep in essential_deps:
                check_script += f"try:\n    import {dep}\nexcept ImportError:\n    print(\'{dep}\')\n"
            
            # Write check script to a temporary file
            temp_script_path = os.path.join(self.script_dir, "temp_check.py")
            with open(temp_script_path, "w") as f:
                f.write(check_script)
            
            # Run the check script in the conda environment
            result = subprocess.run(
                [self.conda_executable, "run", "-n", self.env_name, "python", temp_script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            # Remove the temporary script
            try:
                os.remove(temp_script_path)
            except:
                pass
            
            if result.returncode == 0:
                missing_deps = [dep for dep in result.stdout.strip().split("\n") if dep]
                
                if missing_deps:
                    missing_deps_str = ' '.join(missing_deps)
                    logger.warning(f"The following essential dependencies are missing: {missing_deps_str}")
                    return missing_deps
                else:
                    logger.info("All essential dependencies are installed.")
                    return []
            else:
                logger.error(f"Error checking missing dependencies: {result.stderr.strip()}")
                return essential_deps # Assume all are missing if check script fails
                
        except Exception as e:
            logger.error(f"Error checking missing dependencies: {str(e)}")
            return essential_deps # Assume all are missing if an error occurs

    def run_setup(self):
        overall_success = True

        if not self.check_conda_installed():
            overall_success = False
            logger.error("Setup cannot proceed without Conda.")
            return False

        if not self.create_conda_env():
            overall_success = False
            logger.error("Failed to create or activate conda environment.")

        if not self.install_pytorch():
            overall_success = False
            logger.error("Failed to install PyTorch.")

        if not self.install_nnunetv2():
            overall_success = False
            logger.error("Failed to install nnUNetV2.")

        if not self.install_requirements_txt():
            overall_success = False
            logger.error("Failed to install dependencies from requirements.txt.")

        # Graphviz is now handled with a download attempt
        if not self.download_and_setup_graphviz():
            overall_success = False
            logger.warning("Graphviz setup might require manual intervention. Please check logs.")
        else:
            self.setup_graphviz_path()

        missing = self.check_missing_dependencies()
        if missing:
            overall_success = False
            logger.error("Setup completed with missing dependencies: " + " ".join(missing))
        
        if not self.validate_installation():
            overall_success = False
            logger.error("Installation validation failed. Please review the logs.")
        
        if overall_success:
            logger.info("nnUNetV2 setup completed successfully!")
        else:
            logger.error("nnUNetV2 setup completed with some errors. Please review the logs above for details.")
        
        return overall_success

def main():
    parser = argparse.ArgumentParser(description="nnUNetV2 Setup Script")
    parser.add_argument("--env_name", type=str, default="nnunetv2", help="Name of the conda environment")
    parser.add_argument("--cuda_version", type=str, default="11.7", help="CUDA version for PyTorch (e.g., 11.7, 11.8, 12.1)")
    parser.add_argument("--python_version", type=str, default="3.12", help="Python version for the conda environment")
    args = parser.parse_args()

    setup_manager = CondaSetup(env_name=args.env_name, cuda_version=args.cuda_version, python_version=args.python_version)
    setup_manager.run_setup()

if __name__ == "__main__":
    main()


