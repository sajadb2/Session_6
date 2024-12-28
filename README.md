![GitHub repo size](https://img.shields.io/github/repo-size/sajadb2/Assignments_EV3)
![GitHub contributors](https://img.shields.io/github/contributors/sajadb2/Assignments_EV3)
![GitHub stars](https://img.shields.io/github/stars/sajadb2/Assignments_EV3?style=social)
![GitHub forks](https://img.shields.io/github/forks/sajadb2/Assignments_EV3?style=social)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
![Tests](https://github.com/sajadb2/Assignments_EV3/actions/workflows/python-app.yml/badge.svg)

# EVA4 Session 6 Assignment

This repository contains the implementation of a Convolutional Neural Network (CNN) for the MNIST dataset using PyTorch.

## Project Structure

## Model Architecture

The model consists of the following layers:
- Input Layer: 28x28x1
- Convolution layers with batch normalization and dropout
- Global Average Pooling (GAP)
- Output Layer: 10 classes

### Network Structure

## Training Results

- Best Training Accuracy: 99.35%
- Best Test Accuracy: 99.42%
- Number of Parameters: 15,088

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib
- tqdm
- pytest (for testing)
- flake8 (for linting)

## Installation

1. Clone the repository:

## Usage

To train the model:

## Model Features

- Uses Batch Normalization
- Implements Dropout for regularization
- Global Average Pooling (GAP)
- Less than 20k parameters
- Achieves >99.4% accuracy

## Testing

The project includes unit tests that verify:
- Model output shape
- Parameter count (<20k)
- Forward pass functionality

Tests are automatically run on every push and pull request through GitHub Actions.

## Continuous Integration

This project uses GitHub Actions for:
- Running unit tests
- Code linting with flake8
- Notebook execution verification

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
  None
