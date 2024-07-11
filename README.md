# Handwritten Digit Recognition

This repository contains an application for recognizing handwritten digits using a LeNet model. The application features a graphical user interface (GUI) built with Tkinter, where users can draw digits and extract predictions from the trained model.

## Features

- **Draw Digits:** Use mouse input to Create digits on a canvas.
- **Extract Predictions:** Get the predicted digit and its confidence percentage.
- **Clear Canvas:** Erase the drawing to start over.

## Installation

To set up the project, clone the repository and install the required packages.

```bash
git clone https://github.com/eshal26/Handwritten-Multi-Digit-Classification
cd Handwritten-Multi-Digit-Classification
pip install -r requirements.txt
```
## Model Details

- **Model Architecture:** LeNet is one of the early convolutional neural networks (CNNs) and is a foundational architecture in deep learning for image classification tasks.
  
- **Dataset:** I used the MNIST handwritten digits dataset consisting of 60,000 train images and 10,000 test images. You can import the dataset from torchvision.datasets.MNIST
  
- **Model File:** `lenet_mnist_model.pth`
  
- **Performance:** The model achieves high accuracy (98%) in recognizing handwritten digits from the MNIST dataset.

## Dependencies

The project requires the following Python packages:

- `torch`
- `numpy`
- `opencv-python`
- `Pillow`
- `tkinter`

You can install these dependencies using the provided `requirements.txt` file:

```text
torch
numpy
opencv-python
Pillow
```
## Contributing
Contributions are welcome! Please submit a pull request describing your changes and any relevant tests. If you find bugs or have suggestions, feel free to open an issue.

## Screenshot

![image](https://github.com/eshal26/Handwritten-Multi-Digit-Classification/assets/124394813/117c3963-013e-4434-a8e4-f0773a1dc7bb)


