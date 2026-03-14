# EMG_NN_AM_Signal_Processing_for_Gesture_Classification

This project, created under the mentorship of AM, is for preprocessing and decoding EMG signals, based on Mikhail Lebedev's papers. The neural network predicts numbers based on 8-channel EMG data.


## Project Structure

* **data_check.ipynb**
Initial Exploratory Data Analysis (EDA). It includes signal visualization using the MNE library and verification of data integrity from .mat files.

* **emg-data-check.ipynb**
Specifically focuses on validating EMG datasets, checking label consistency, and calculating dataset statistics across multiple subjects.

* **preproc.ipynb**
Contains the core model architecture and the training pipeline. It includes custom PyTorch layers for real-time signal preprocessing and data augmentation.

* **preproc-cv.ipynb**
An advanced version of the training pipeline that utilizes Stratified K-Fold Cross-Validation to ensure model stability and generalizability.

## Key Technical Features

### Signal Preprocessing
The project implements a custom EMGPreproc module that integrates directly into the neural network as a layer. This allows the model to learn optimal filtering parameters.
* **FIR Filtering**: Finite Impulse Response filters are used for envelope extraction.
* **Trainable Decimation**: The system can learn how to downsample signals effectively while preserving critical features.

### Data Augmentation
To prevent overfitting, a specialized EMGAugment class is used, providing the following transformations:
* **Additive Noise**: Gaussian noise injection.
* **Signal Scaling**: Random gain and offset adjustments.
* **Temporal Transformations**: Time shifting and time masking.
* **Channel Manipulation**: Channel dropout and mixing of adjacent channels.

## Requirements

The project is developed using Python 3.12. The following libraries are required:
* PyTorch
* MNE
* SciPy
* Scikit-learn
* Matplotlib/Seaborn

## Usage

1. **Environment Setup**: Install the required dependencies using pip.
2. **Data Inspection**: Run data_check.ipynb to visualize raw signals.
3. **Model Training**: Execute preproc-cv.ipynb to start the training process with cross-validation.
