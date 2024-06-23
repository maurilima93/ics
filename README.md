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

2. Install the required dependencies:
   ```bash
   pip install tensorflow keras numpy

3. Ensure you have the descriptions.txt file in the project directory. This file should contain the image IDs and corresponding captions.

4. Prepare your image dataset. Place your images in a directory and update the image_paths list in the ics.py script with the correct paths to your images.

5. Usage - Extract image features using the pre-trained VGG16 model and prepare the training sequences:
   ```bash
   python ics.py

## Acknowledgements
This project utilizes the COCO dataset for training and evaluation. Special thanks to the developers of TensorFlow and Keras for their excellent deep learning libraries.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
