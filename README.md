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



