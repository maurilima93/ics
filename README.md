# Image Captioning System

This project is an end-to-end image captioning system that generates descriptive captions for images using Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. The system achieves state-of-the-art performance on the COCO dataset.

## Overview

The image captioning system involves two main components:
1. **Feature Extraction**: Using a pre-trained VGG16 model to extract features from images.
2. **Caption Generation**: Using LSTM networks to generate captions based on the extracted image features.

## Project Structure

- `ics.py`: The main script containing the implementation of the image captioning system.
- `descriptions.txt`: A file containing image descriptions (captions) for training.

## Prerequisites

- Python 3.x
- TensorFlow
- Keras
- NumPy

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/maurilima93/ics.git
   cd ics
