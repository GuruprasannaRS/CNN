# NeuralNet Toolkit: CNN, RNN & NLP Implementations

This repository contains implementations of various neural network architectures for image classification and other machine learning tasks. The project includes CNN models for CIFAR-10 image classification, RNN models for stock price prediction, and NLP models for sentiment analysis.

## Project Structure

```
CNN/
├── cnn_image_classification.ipynb    # Main CNN implementation for CIFAR-10
├── train_RNN.py                      # RNN training script for stock prediction
├── test_RNN.py                       # RNN testing script
├── train_NLP.py                      # NLP training script for sentiment analysis
├── test_NLP.py                       # NLP testing script
├── data/                             # Dataset directory
│   ├── aclImdb/                      # IMDB dataset for sentiment analysis
│   ├── q2_dataset.csv               # Stock price dataset
│   ├── train_data_RNN.csv           # Processed RNN training data
│   └── test_data_RNN.csv            # Processed RNN testing data
└── README.md                         # This file
```

## Problem 1: CNN for Image Classification

### Overview
Implementation of convolutional neural networks for classifying images from the CIFAR-10 dataset. The project uses TensorFlow/Keras and implements multiple MLP and CNN architectures.

### Dataset
- **CIFAR-10**: 60,000 32x32 color images in 10 classes
- Training set: 50,000 images (20% randomly sampled = 10,000 images for training)
- Test set: 10,000 images (used for validation)

### Implemented Models

#### MLP Models
1. **MLP 1**: 2 hidden layers (512, 512) with sigmoid activation
2. **MLP 2**: 5 hidden layers (512, 256, 128, 64, 32) with sigmoid activation
3. **MLP 3**: 3 hidden layers (256, 256, 256) with sigmoid activation
4. **MLP 4**: 3 hidden layers (1024, 512, 256) with sigmoid activation
5. **MLP 5**: 4 hidden layers (1024, 512, 256, 256) with sigmoid activation

#### CNN Models
1. **CNN 1**: 2 Conv2D layers (64 filters, 3x3) + 2 Dense layers (512, 512)
2. **CNN 2**: 2 Conv2D layers (64 filters, 3x3) + MaxPooling2D + 2 Dense layers (512, 512) + Dropout
3. **CNN 3**: 4 Conv2D layers (64, 64, 128, 128) + MaxPooling2D + 2 Dense layers (128, 128) + Dropout

### Training Parameters
- Batch size: 32
- Epochs: 5 (initial), 10 (extended training)
- Optimizer: Adam
- Loss function: Categorical crossentropy
- Metrics: Accuracy

### Results Summary
- **Best MLP**: MLP 4 achieved highest validation accuracy
- **Best CNN**: CNN 3 achieved 57.99% validation accuracy
- **Overfitting Analysis**: Extended training showed overfitting in CNN models

## Problem 2: RNN for Stock Price Prediction

### Overview
LSTM-based recurrent neural network for predicting stock opening prices using historical market data.

### Dataset
- **Features**: Open, High, Low, Volume prices
- **Target**: Next day's opening price
- **Time window**: 3 days of historical data
- **Data split**: 70% training, 30% testing

### Model Architecture
- **LSTM layers**: 3 layers with 256 units each
- **Dropout**: 0.2 dropout rate between LSTM layers
- **Dense layer**: 1 output unit for price prediction
- **Activation**: Linear (default)

### Data Processing
- Min-Max scaling (0-1 range)
- Reshaped to (samples, timesteps, features) format
- Random shuffling with seed 42

## Problem 3: NLP for Sentiment Analysis

### Overview
Convolutional neural network for sentiment analysis on IMDB movie reviews dataset.

### Dataset
- **IMDB Dataset**: Movie reviews with positive/negative labels
- **Text preprocessing**: HTML removal, special character cleaning, lowercase conversion
- **Sequence length**: Maximum 1000 tokens

### Model Architecture
- **Embedding layer**: 16-dimensional word vectors
- **Conv1D**: 16 filters with kernel size 2
- **GlobalAveragePooling1D**: Sequence pooling
- **Dense layers**: 2 hidden layers (16, 16) with ReLU activation
- **Dropout**: 0.2 dropout rate throughout
- **Output**: 1 unit with sigmoid activation for binary classification

### Training Parameters
- **Optimizer**: Adam
- **Loss function**: Binary crossentropy
- **Metrics**: Accuracy
- **Validation split**: 30%

## Usage

### CNN Image Classification
```bash
# Open the Jupyter notebook
jupyter notebook cnn_image_classification.ipynb
```

### RNN Stock Prediction
```bash
# Training
python train_RNN.py

# Testing
python test_RNN.py
```

### NLP Sentiment Analysis
```bash
# Training
python train_NLP.py

# Testing
python test_NLP.py
```

## Dependencies

- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Key Findings

1. **MLP Performance**: Deeper networks with more layers showed decreased performance due to overfitting
2. **CNN Advantage**: CNN models consistently outperformed MLP models for image classification
3. **Overfitting**: Extended training epochs led to overfitting in CNN models
4. **Architecture Impact**: Adding dropout and pooling layers improved CNN generalization
5. **Data Sampling**: Using 20% of training data showed the importance of sufficient training samples

## Model Performance Comparison

| Model Type | Best Validation Accuracy |
|------------|-------------------------|
| MLP 4      | 38.25%                 |
| CNN 1      | 54.65%                 |
| CNN 2      | 55.41%                 |
| CNN 3      | 57.99%                 |

## Notes

- All models use the same training parameters for fair comparison
- Results are based on 5 epochs of training unless specified otherwise
- The project demonstrates the effectiveness of CNNs over MLPs for image classification tasks
- RNN and NLP implementations show practical applications of neural networks in different domains