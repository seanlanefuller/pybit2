# pybit2

A character-level neural network for sequence prediction and generation, built from scratch using NumPy.

## Overview

`pybit2` implements a multi-layer perceptron (MLP) designed for text-based tasks. It learns patterns from a training dataset (like a story or a set of examples) and can then predict the next character in a sequence or generate entire passages based on a given prompt.

## Features

- **Custom Neural Network**: A feed-forward architecture with three hidden layers and sigmoid activation.
- **Dropout Regularization**: Prevents overfitting during training by randomly dropping units.
- **Autoregressive Generation**: Generates sequences character-by-character based on a sliding window of previous context.
- **Weight Management**: Built-in functionality to save and load model weights from `.npz` files.
- **Character Encoding**: Uses one-hot encoding for a custom-built vocabulary derived from the training data.

## Installation

Ensure you have Python 3 installed. You will also need NumPy:

```bash
pip install numpy
```

## Usage

### Training and Generation

The main entry point is `pybit.py`. It loads training data from `training_data.txt`, trains the model (or loads existing weights from `model_weights.npz`), and then performs predictions and generates sample text.

```bash
python pybit.py
```

### Files

- `pybit.py`: The core implementation of the neural network and training loop.
- `training_data.txt`: The text file used for training the model.
- `model_weights.npz`: Compressed NumPy file storing the trained model weights.
- `run.bat`: A simple batch script to execute the Python script (Windows).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
