Scene Text Recognition System

## Overview
This project is a comprehensive scene text recognition (STR) system that integrates PyQt for the user interface, YOLOv8 for text detection, and PaddleOCR for multilingual text recognition. It supports efficient image preprocessing, CUDA-accelerated recognition, and flexible text region selection, suitable for various applications like document digitization, autonomous driving, and digital content processing.

## Features
- **User-Friendly Interface**: Built with PyQt, providing an intuitive platform for users to interact with STR functionalities.
- **Efficient Text Detection**: YOLOv8 model identifies text regions accurately, even in complex scenes.
- **Multilingual OCR**: PaddleOCR supports multiple languages, enhancing system versatility across different use cases.
- **CUDA Acceleration**: Utilizes CUDA if available for faster processing, with automatic fallback to CPU mode.
- **Modular and Extendable**: The system architecture is modular, allowing for easy updates and future enhancements.

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional but recommended for acceleration)
- Git for cloning the repository

### Step 1: Clone the Repository
First, clone this repository using Git:
```bash
git clone https://github.com/White0000/Yolov8_Transfomer_Pytorch.git
```

### Step 2: Set Up Virtual Environment

Set up a virtual environment to keep your dependencies isolated:
```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

### Step 3: Install Dependencies
Ensure you have all required dependencies by installing from `requirements.txt`:
```bash
pip install -r requirements.txt
```
If you encounter issues related to GPU support, ensure you have the correct CUDA and cuDNN versions installed, matching your PyTorch setup.

### Step 4: Verify Environment
Run the environment test script to verify that your system is set up correctly:
```bash
python envirtest.py
```
This script checks for CUDA availability and ensures that the required libraries (e.g., PyTorch, OpenCV, PyQt5) are correctly installed.

## Usage

### 1. Run the Application
To launch the application, use:
```bash
python main.py
```
This will start the GUI, where you can load images and perform text recognition tasks.

### 2. Load an Image
- Click the **"Select Image"** button to load an image (PNG, JPG, BMP formats supported).
- The image will be displayed in the interface.

### 3. Preprocess the Image
- Use the **"Grayscale"**, **"Binarize"**, and **"Denoise"** buttons to preprocess the image. These options enhance the text visibility and prepare the image for recognition.
- Buttons are enabled sequentially to guide you through the correct processing steps.

### 4. Select Text Region and Recognize
- Click **"Select Area"** to manually select the text region using a drag-and-drop interface. This helps focus the recognition on relevant portions of the image.
- Click **"Start Recognition"** to process the selected area. The recognized text will appear in the text box.

### 5. Export Recognized Text
- Use the **"Export as TXT"** button to save the recognized text to a file.

## Training and Testing

### Training the YOLOv8 Model
To train the YOLOv8 model with the dataset:
1. Place your dataset in the appropriate folder and update the paths in `data.yaml`.
2. Run the training script:
   ```bash
   python trainv8.py
   ```
   - The script uses the configuration from `data.yaml` and saves the trained model in the specified output path.

### Testing the Trained Model
To evaluate the trained model:
1. Run the testing script:
   ```bash
   python processv8train.py
   ```
   - This script tests the model on the validation set and outputs performance metrics like precision, recall, and F1-score.

## Dataset Preparation
- The system uses the **IIIT5K** dataset by default. Ensure the dataset is structured as specified in `data.yaml`:
  - Training images should be placed in `dataset/train`.
  - Validation images should be placed in `dataset/val`.
- Update `data.yaml` if you use a custom dataset, ensuring paths and labels are correctly set.

## System Requirements
- **Python**: Version 3.8+
- **Dependencies**: PyTorch, OpenCV, PyQt5, PaddleOCR, and others (see `requirements.txt`).
- **Hardware**: 
  - GPU with CUDA support (recommended for faster training and inference).
  - CPU mode is supported but may be slower.

## Directory Structure
Here’s a breakdown of the key files and scripts in the repository:

- **`main.py`**: Main file to launch the GUI and start the application.
- **`trainv8.py`**: Script for training the YOLOv8 model with specified datasets.
- **`processv8train.py`**: Script to test and validate the trained model’s performance.
- **`data.yaml`**: Configuration file for specifying dataset paths and labels.
- **`crnn_model.py`**: Implementation of the CRNN model for sequence-based text recognition tasks.
- **`envirtest.py`**: Script to verify system setup and check CUDA availability.

## Advanced Usage
### Fine-Tuning and Customization
- The modular nature of the code allows for fine-tuning. You can replace the YOLOv8 model with another variant or add new preprocessing steps in the GUI.
- Modify `crnn_model.py` for specialized sequence recognition tasks or to integrate additional features.

### Adding New Datasets
- Place your new dataset under the `dataset/` directory.
- Update `data.yaml` with the new dataset paths and labels.
- Retrain the model using the `trainv8.py` script.

## Future Work
- Add support for real-time video stream processing.
- Expand language support in PaddleOCR for better multilingual capabilities.
- Implement more advanced image preprocessing methods for improved accuracy on low-contrast and distorted text.

## Support
For any issues or inquiries, please open an issue on GitHub or contact the project maintainer directly.
