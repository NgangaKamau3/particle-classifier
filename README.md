# Electron-Photon Classifier

A deep learning model for classifying electrons and photons using ResNet-15 architecture.

Description: 32x32 matrices with two channels: hit energy and time for two types of
particles, electrons and photons, hitting the detector.

The project uses a Resnet-15 like architecture in PyTorch to classify between photons and electrons based on the above criteria, that is, hit energy and time as recorded by the detector.


## Project Structure

```
particle-classifier/
├── notebooks/          # Jupyter notebooks
├── src/               # Source code
└── data/              # Dataset directory
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

See `requirements.txt` for full dependencies.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/NgangaKamau3/particle-classifier.git
cd particle-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset files and place them in the `data/` directory.

## Usage

### Using Python Script
```bash
python src/classifier.py
```

### Using Jupyter Notebook
Open `notebooks/classifier.ipynb` in Jupyter Lab/Notebook.

## Dataset

The dataset consists of two HDF5 files:
- `SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5`
- `SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5`

Each file contains:
- Energy channel
- Time channel

## Model Architecture

ResNet-15 architecture with:
- 2 input channels (energy and time)
- 6 residual blocks
- Dropout regularization
- Batch normalization

# Data Matrix Specifications

## Input Matrix
- **Shape**: (batch_size, channels, height, width)
  - batch_size: Variable (128 in training)
  - channels: 2 (energy and time)
  - height: 32
  - width: 32
- **Data Type**: torch.float32
- **Value Range**: Normalized between 0 and 1
- **Example shape**: (128, 2, 32, 32)

## Output Matrix
- **Shape**: (batch_size, num_classes)
  - batch_size: Same as input
  - num_classes: 2 (electron=0, photon=1)
- **Data Type**: torch.float32
- **Value Range**: [0,1] after softmax
- **Example shape**: (128, 2)

## Data Transformations
Input data goes through these transformations:
1. HDF5 shape (N, 32, 32, 2) → PyTorch shape (N, 2, 32, 32)
2. Data type conversion: numpy.ndarray → torch.FloatTensor
3. Device transfer: CPU → GPU (if available)



