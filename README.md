# UNET_from_Scratch
Here's a README file for the Python code which implements a U-Net model using PyTorch. This README provides a comprehensive overview of the U-Net model implementation, including its architecture, setup, and a basic example of how to run the model.

---

# U-Net Model Implementation in PyTorch

This Python module implements a U-Net architecture for image segmentation tasks using PyTorch. U-Net is widely used in biomedical image segmentation because of its efficient use of GPU memory and its ability to work with very few training samples.

## Model Components

The U-Net model is structured into a contracting path to capture context and a symmetric expanding path that enables precise localization. Here are the key components and layers included in the implementation:

- **Double Convolution (`doub_conv`)**: A function that applies two consecutive sets of convolution followed by a ReLU activation function. This function is used multiple times to build both the downsampling and upsampling parts of the network.

- **Convolution Transpose (`conv_transpose`)**: A function to perform upsampling in the expansive path using transposed convolutions.

- **Max Pooling**: Applied after each double convolution block in the downsampling path to reduce the spatial dimensions.

- **Center Crop**: Used in the expansive path to crop the feature maps from the contracting path before concatenation. This ensures that the feature maps from the contracting path and the expansive path are aligned by size.

- **Sequential Model Building**: The U-Net architecture is constructed using a sequence of layers stored as class attributes, including the downsampling, bottleneck, and upsampling layers.

## Usage

To use this U-Net implementation, instantiate the `UNetModel` class and pass an input tensor of the appropriate size (572x572 pixels with 1 channel for grayscale images, in this example). Below is a simple example of how to create an instance of the model and pass a single image through it:

```python
import torch
from model import UNetModel

# Create a random single-channel image of shape (1, 1, 572, 572)
image = torch.rand((1, 1, 572, 572))

# Instantiate the U-Net model
model = UNetModel()

# Perform a forward pass through the model
output = model(image)

# Print the output shape
print(output.shape)  # Expect shape to be (1, 2, 388, 388) for 2 classes
```

## Installation

To use this module, you must have Python and PyTorch installed. The code is compatible with PyTorch 1.x versions. If you do not have PyTorch installed, you can install it using pip:

```bash
pip install torch torchvision
```

