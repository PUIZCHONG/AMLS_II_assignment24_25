# Cassava Leaf Disease Classification

This repository contains the source code for a machine learning project focused on cassava leaf disease classification using ensemble techniques with CropNet and EfficientNetB4 models.

## Overview

The project aims to classify cassava leaf images into different disease categories using deep learning models. It employs an ensemble approach combining predictions from CropNet and EfficientNetB4 for improved accuracy.

## Model Download

**Note:** The trained models are too large to be uploaded to GitHub. Please download them from Kaggle using the links below:

- **CropNet Model:** [Download from Kaggle](https://www.kaggle.com/code/nocharon/cropnet) 
- **EfficientNetB4 Model:** [Download from Kaggle](https://www.kaggle.com/code/nocharon/cassava-2)

After downloading, place the model files in the root directory of this repository.

## Repository Structure

### Source Files
- `data.py`: Analyzes the cassava dataset, including RGB value distribution, class distribution, and visualization of various image augmentation techniques.
- `trained_cropnet.py`: Implements Grad-CAM visualization for the CropNet model to highlight regions of interest in classification decisions.
- `trained_efficientnet.py`: Implements Grad-CAM visualization for the EfficientNetB4 model.
- `esemble.py`: Combines outputs from both models to produce ensemble predictions with improved accuracy.
- `main.py`: Orchestrates the execution of all components for a complete pipeline run.

### Notebook Folder
- `check_data.ipynb`: Data analysis notebook for finding the mean and standard deviation of the dataset.
- `cropnet.ipynb`: Training code for the CropNet model, configured to run on Kaggle GPU P100.
- `efficientnetb4.ipynb`: Training code for the EfficientNetB4 model, configured to run on Kaggle GPU P100.

## Requirements

```
tensorflow>=2.4.0
pytorch>=1.7.0
numpy
pandas
matplotlib
opencv-python
scikit-learn
```

## Usage

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/cassava-leaf-disease-classification.git
   cd cassava-leaf-disease-classification
   ```

2. Download the models from Kaggle using the links provided above.

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the full pipeline:
   ```
   python main.py
   ```

## Dataset Analysis

The `data.py` script performs the following analyses on the cassava leaf dataset:
- RGB value distribution across disease classes
- Count of images per disease class
- Visualization of image augmentation techniques (rotation, flipping, etc.)

## Grad-CAM Visualization

The `trained_cropnet.py` and `trained_efficientnet.py` scripts generate Grad-CAM visualizations that highlight the regions of the leaf images that most influenced the classification decision, providing interpretability to the model predictions.

## Ensemble Model

The `esemble.py` script combines predictions from both models using a weighted averaging approach to improve overall classification accuracy.

## License

[Add your license information here]

## Acknowledgements

[Add any acknowledgements or references here]