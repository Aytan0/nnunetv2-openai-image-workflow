# 🧠 nnUNet Automation Bot

Automated workflow for medical image segmentation using nnUNet framework with AI-powered dataset organization.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![nnUNet](https://img.shields.io/badge/nnUNet-v2.0+-green.svg)](https://github.com/MIC-DKFZ/nnUNet)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📋 Features

- ✅ **Automated Setup**: One-command installation and configuration
- 🤖 **AI-Powered Organization**: Automatic dataset structuring using OpenAI GPT
- 📊 **Dataset Management**: Easy dataset creation and conversion
- 🚀 **Model Training**: Simplified nnUNet training workflow
- 📈 **Visualization**: Built-in result visualization tools
- 🔄 **Synapse Integration**: Direct download from Synapse.org

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended)
- 10GB+ free disk space
- Anaconda or Miniconda (optional)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Aytan0/nnunetv2-openai-image-workflow.git
cd automation-bot
```

2. **Run automated setup**
```bash
python setup_optimized.py
```

3. **Activate environment**
```bash
# If using conda
conda activate nnunetv2

# Or use directly
python run_nnunet.py
```

### Manual Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup nnUNet environment
python -m src.setup_environment
```

## 📖 Usage

### Interactive Menu

```bash
python run_nnunet.py
```

**Available Options:**
1. Setup Environment Variables
2. Download from Synapse
3. Automatic Dataset Selection
4. Train Model
5. Make Predictions
6. Visualize Results
7. Clear Data
8. Upload to Synapse
9. **AI-Powered Data Organization** (NEW!)
10. List Existing Datasets
11. Exit

### AI-Powered Dataset Organization

Place your zip files in `data/raw/Dataset_XXX/` and run:

```bash
# Select option 9 from menu
python run_nnunet.py
```

The AI will:
- Extract and analyze your dataset structure
- Suggest optimal nnUNet organization
- Automatically reorganize files
- Create dataset.json

### Command Line Usage

```bash
# Download dataset
python main.py download --synapse-id syn12345 --username your@email.com

# Convert dataset
python main.py convert --task-name MyTask --task-id 001

# Train model
python main.py train --dataset 001 --fold 0

# Make predictions
python main.py predict --dataset 001 --input-folder /path/to/test
```

## 📁 Project Structure

```
automation-bot/
├── src/
│   ├── __init__.py
│   ├── workflow.py          # Main workflow manager
│   ├── ai_organizer.py      # AI-powered dataset organization
│   ├── dataset_manager.py   # Dataset handling
│   ├── synapse_download.py  # Synapse integration
│   └── setup_environment.py # Environment setup
├── data/
│   ├── raw/                 # Raw zip files
│   └── nnUNet/
│       ├── nnUNet_raw/      # Processed datasets
│       ├── nnUNet_preprocessed/
│       └── nnUNet_results/
├── logs/                    # Log files
├── main.py                  # CLI entry point
├── run_nnunet.py           # Simplified runner
├── setup_optimized.py      # Automated setup
├── requirements.txt        # Dependencies
├── .env.example           # Environment template
└── README.md

```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:

```bash
# nnUNet paths
nnUNet_raw=C:/path/to/data/nnUNet/nnUNet_raw
nnUNet_preprocessed=C:/path/to/data/nnUNet/nnUNet_preprocessed
nnUNet_results=C:/path/to/data/nnUNet/nnUNet_results

# Synapse credentials
SYNAPSE_USERNAME=your@email.com
SYNAPSE_PASSWORD=your_password

# OpenAI API (for AI organization)
OPENAI_API_KEY=sk-...

# Optional
PYTHONIOENCODING=utf-8
```

## 🤖 AI Organization Details

The AI organizer:
1. Analyzes folder structure
2. Identifies images and labels
3. Suggests nnUNet-compatible organization
4. Creates `dataset.json`
5. Moves files to correct locations

**Supported formats:**
- NIfTI (`.nii`, `.nii.gz`)
- DICOM
- PNG/JPG (2D slices)

## 📊 Dataset Format

nnUNet expects this structure:

```
Dataset001_TaskName/
├── imagesTr/
│   ├── case_0000_0000.nii.gz
│   └── case_0001_0000.nii.gz
├── labelsTr/
│   ├── case_0000.nii.gz
│   └── case_0001.nii.gz
├── imagesTs/  (optional)
└── dataset.json
```

## 🔧 Troubleshooting

### Common Issues

**Unicode Errors:**
```bash
# Set encoding
export PYTHONIOENCODING=utf-8  # Linux/Mac
set PYTHONIOENCODING=utf-8     # Windows
```

**Import Errors:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**CUDA Not Found:**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

## 📝 Examples

### Example 1: Medical Imaging Dataset

```bash
# Place data in data/raw/Dataset_001/
# Run AI organization
python run_nnunet.py
# Select option 9
# Follow prompts
```

### Example 2: Training from Scratch

```bash
# Setup
python setup_optimized.py

# Organize data
python run_nnunet.py  # Option 9

# Train
python run_nnunet.py  # Option 4

# Predict
python run_nnunet.py  # Option 5
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [nnUNet](https://github.com/MIC-DKFZ/nnUNet) - Medical Image Segmentation Framework
- [Synapse](https://www.synapse.org/) - Data Repository
- [OpenAI](https://openai.com/) - GPT API for AI organization

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 🔗 Links

- [nnUNet Documentation](https://github.com/MIC-DKFZ/nnUNet)
- [nnUNet Paper](https://www.nature.com/articles/s41592-020-01008-z)

---

**⚠️ Note:** This tool is for research purposes. Always validate results for clinical applications.
