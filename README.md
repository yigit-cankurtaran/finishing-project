# EEG/ECG Time Series Analysis and Classification

A comprehensive machine learning project for analyzing and classifying electroencephalography (EEG) and electrocardiography (ECG) time series data using deep learning techniques.

## Project Overview

This project implements neural network models for the classification of biomedical signals:
- **EEG Classification**: Binary classification of brain signals (normal vs abnormal)
- **ECG Classification**: Multi-label classification of heart conditions using the PTB-XL dataset

## Project Structure

```
├── eeg_training.py          # EEG model training with 1D CNN
├── eeg_inference.py         # EEG inference script
├── eeg_ripper.py           # EEG data preprocessing utilities
├── ecg_training.py         # Basic ECG training implementation
├── enhanced_ecg_train.py   # Advanced ECG training with ResNet architecture
├── ecg_inference.py        # ECG inference and visualization
├── device_func.py          # Multi-platform device selection (CPU/GPU/MPS)
├── projectphotos/          # Training screenshots and results
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Datasets

### EEG Dataset
- **Source**: [MTOUH EEG Dataset](https://www.kaggle.com/datasets/buraktaci/mtouh?resource=download)
- Uses `.mat` files containing multi-channel EEG recordings
- 35 channels, 7500 samples per epoch (15 seconds at 500Hz)
- Binary classification: Normal (0) vs Abnormal (1)
- Labels recoded from original [1,11] → 1, others → 0

### ECG Dataset (PTB-XL)
- **Source**: [PTB-XL Dataset](https://physionet.org/content/ptb-xl/1.0.3/)
- Large publicly available electrocardiography dataset
- 12-lead ECG recordings at 500Hz sampling rate
- Multi-label classification across diagnostic classes
- SCP (Systematic Coronary Risk Evaluation) code mapping for heart conditions

![Dataset Structure](projectphotos/Screenshot%202025-01-11%20at%2000.27.33.png)

## Model Architectures

### EEG Model (1D CNN)
- **Input**: 35 channels × 7500 timesteps
- **Architecture**: 
  - Conv1d(35→64, k=7) + BatchNorm + ReLU + MaxPool(2)
  - Conv1d(64→128, k=5) + BatchNorm + ReLU + MaxPool(2)  
  - Conv1d(128→256, k=3) + BatchNorm + ReLU + AdaptiveAvgPool
  - Linear(256→2) for binary classification
- **Training**: Adam optimizer, early stopping (patience=5)

### ECG Model (ResNet-based)
- **Input**: 12 channels × 5000 timesteps
- **Architecture**: ResNet blocks with residual connections
  - Stem: Conv1d(12→64, k=7) + BatchNorm + ReLU
  - 2× ResBlocks with skip connections
  - Global Average Pooling + Linear classification head
- **Features**: 
  - Multi-label classification capability
  - Weighted loss for class imbalance
  - Data augmentation (noise, shift, drift)
  - Mixed precision training

![Training Output](projectphotos/Screenshot%202025-06-19%20at%2001.08.24.png)

## Signal Processing Pipeline

### Preprocessing Steps
1. **Notch Filtering**: Remove 50Hz powerline interference
2. **Bandpass Filtering**: 
   - EEG: 1-40Hz
   - ECG: 0.5-40Hz
3. **Z-score Normalization**: Per-channel standardization
4. **Window Segmentation**: Fixed-size temporal windows

### Data Augmentation (ECG)
- Gaussian noise addition (σ=0.005)
- Temporal shifting (±100 samples)
- Baseline drift simulation

## Training Results

The models demonstrate strong performance on their respective tasks:

### EEG Classification
- Binary classification accuracy: ~95%+
- Early stopping prevents overfitting
- Real-time inference capability

### ECG Classification  
- Multi-label F1 scores vary by diagnostic class
- Handles class imbalance through weighted loss
- Supports multiple simultaneous heart conditions

![ECG Predictions](projectphotos/Screenshot%202025-06-19%20at%2001.09.53.png)

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### EEG Training
```bash
python eeg_training.py
```

### ECG Training (Enhanced)
```bash
python enhanced_ecg_train.py \
  --csv ./ptbxl_database.csv \
  --signals ./signals \
  --scp ./scp_statements.csv
```

### Inference

#### EEG Inference
```bash
python eeg_inference.py epoch_data.npy [model_checkpoint.pth]
```

#### ECG Inference  
```bash
python ecg_inference.py \
  --record path/to/ecg/record \
  --model ecgnet_best.pth \
  --plot
```

![Code Example](projectphotos/Screenshot%202025-06-19%20at%2001.13.52.png)

## Technical Features

- **Cross-platform GPU Support**: Automatic detection of CUDA, MPS (Apple Silicon), or CPU
- **Mixed Precision Training**: Faster training with reduced memory usage
- **Early Stopping**: Prevents overfitting with patience-based monitoring
- **Model Checkpointing**: Saves best models based on validation metrics
- **Data Visualization**: Built-in plotting for signals and predictions
- **Modular Design**: Separate training and inference pipelines

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, SciPy, Pandas
- scikit-learn, scikit-multilearn
- WFDB (for ECG data)
- Matplotlib (for visualization)

See `requirements.txt` for complete dependency list.

## Model Performance

### Key Metrics
- **EEG**: Binary accuracy, confidence scores
- **ECG**: Macro F1-score, per-class precision/recall
- **Real-time**: Both models support real-time inference

### Hardware Compatibility
- CPU: Full functionality
- NVIDIA GPU: CUDA acceleration
- Apple Silicon: MPS acceleration
- Automatic batch size adjustment for memory constraints

## Future Work

- Integration of transformer architectures
- Multi-modal EEG+ECG analysis  
- Real-time streaming data processing
- Clinical validation studies
- Mobile deployment optimization

---

*This project demonstrates the application of deep learning to biomedical signal analysis, providing robust tools for EEG and ECG classification tasks.* 
