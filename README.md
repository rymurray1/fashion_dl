# Fashion MNIST Deep Learning Classification Project

## Overview

This project implements a Convolutional Neural Network (CNN) to classify fashion items from the Fashion MNIST dataset. The model achieves approximately **89.5% accuracy** on the test set, demonstrating the effectiveness of CNNs for image classification tasks.

## Problem Statement

Fashion MNIST is a dataset of Zalando's article images consisting of 28x28 grayscale images in 10 categories. The task is to classify each image into its corresponding fashion category using deep learning techniques.

**Real-world applications:**
- E-commerce automation
- Inventory management
- Visual product search
- Retail analytics

## Dataset

The Fashion MNIST dataset contains:
- **60,000 training images**
- **10,000 test images**
- **10 classes** of fashion items:
  - T-shirt/top
  - Trouser
  - Pullover
  - Dress
  - Coat
  - Sandal
  - Shirt
  - Sneaker
  - Bag
  - Ankle boot

Each image is 28x28 pixels in grayscale format.

## Model Architecture

The CNN model consists of:
- **Input Layer**: Reshape to (28, 28, 1) for channel dimension
- **Convolutional Layer 1**: 32 filters, 3x3 kernel, ReLU activation
- **Max Pooling Layer 1**: 2x2 pooling
- **Convolutional Layer 2**: 64 filters, 3x3 kernel, ReLU activation
- **Max Pooling Layer 2**: 2x2 pooling
- **Flatten Layer**: Convert to 1D vector
- **Dense Layer 1**: 64 neurons, ReLU activation
- **Dropout Layer**: 50% dropout for regularization
- **Output Layer**: 10 neurons, softmax activation

## Results

- **Training Accuracy**: ~87.1%
- **Test Accuracy**: ~89.5%
- **Training Time**: ~5 epochs (~6 minutes)

The model shows good generalization with validation accuracy slightly higher than training accuracy, indicating no overfitting.

## Requirements

### Python Dependencies
```
tensorflow>=2.0.0
matplotlib>=3.0.0
numpy>=1.19.0
```

### Installation

1. Clone or download this repository
2. Install the required packages:
   ```bash
   pip install tensorflow matplotlib numpy
   ```
3. Open the Jupyter notebook:
   ```bash
   jupyter notebook fashion_dl.ipynb
   ```

## Usage

1. **Run the notebook**: Execute all cells in `fashion_dl.ipynb`
2. **Data Loading**: The notebook automatically downloads the Fashion MNIST dataset
3. **Training**: The model trains for 5 epochs with progress displayed
4. **Evaluation**: View accuracy plots and final test results

## Key Features

- **Automatic Data Download**: Uses TensorFlow's built-in dataset loader
- **Data Preprocessing**: Normalizes pixel values to [0, 1] range
- **Visualization**: Sample images and training progress plots
- **Model Evaluation**: Confusion matrix and accuracy metrics
- **Clean Architecture**: Simple yet effective CNN design

## Technical Details

### Data Preprocessing
- Pixel normalization: Divide by 255 to scale to [0, 1]
- Reshape: Add channel dimension for CNN input

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Cross-entropy
- **Metrics**: Accuracy
- **Epochs**: 5
- **Batch Size**: Default (32)

### Model Performance
The model achieves strong performance with minimal hyperparameter tuning, demonstrating the effectiveness of CNNs for image classification tasks.

## Future Improvements

1. **Advanced Architectures**: Experiment with ResNet, MobileNet, or EfficientNet
2. **Data Augmentation**: Apply rotation, scaling, and noise to improve generalization
3. **Transfer Learning**: Use pre-trained models on larger datasets
4. **Ensemble Methods**: Combine multiple models for better performance
5. **Hyperparameter Tuning**: Optimize learning rate, batch size, and architecture

## Project Structure

```
final/
├── fashion_dl.ipynb    # Main Jupyter notebook
└── README.md          # This file
```

## Contributing

Feel free to fork this project and experiment with different architectures, hyperparameters, or additional features.

## License

This project is for educational purposes. The Fashion MNIST dataset is available under the MIT License.

## Acknowledgments

- Fashion MNIST dataset by Zalando Research
- TensorFlow/Keras for the deep learning framework
- University of Colorado MSCS program for the educational context 