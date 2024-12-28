![GitHub repo size](https://img.shields.io/github/repo-size/sajadb2/Assignments_EV_)
![GitHub contributors](https://img.shields.io/github/contributors/sajadb2/Assignments_EV_)
![GitHub stars](https://img.shields.io/github/stars/sajadb2/Assignments_EV_?style=social)
![GitHub forks](https://img.shields.io/github/forks/sajadb2/Assignments_EV_?style=social)

# MNIST Digit Classification with PyTorch

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset.

## Overview

The model achieves high accuracy on MNIST digit classification using a modern CNN architecture with regularization techniques like Batch Normalization and Dropout.

## Model Architecture

The model consists of several convolutional layers followed by ReLU activations, max pooling, and fully connected layers. Regularization techniques such as Batch Normalization and Dropout are used to improve generalization.

## Features

- **Architecture**: Custom CNN with multiple convolutional layers
- **Regularization**: BatchNorm, Dropout (0.1)
- **Training**: SGD optimizer with momentum
- **Dataset**: MNIST handwritten digits
- **Performance Monitoring**: Training/validation loss tracking

## Requirements

- Python 3.x
- PyTorch
- torchvision
- tqdm
- matplotlib (for visualizations)

## Usage

1. Open `EVA4_Session_6.ipynb` in Jupyter Notebook.
2. Run all cells to:
   - Load and preprocess MNIST data.
   - Train the model.
   - Evaluate performance.

## Training Parameters

- Batch Size: 128
- Learning Rate: 0.01
- Momentum: 0.9
- Epochs: 13
- Optimizer: SGD

## Results

The model achieves:
- Training accuracy: ~99%
- Test accuracy: ~98%

## Repository Structure