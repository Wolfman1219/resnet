# [![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=1000&color=F7C5B1&background=89D1FF00&center=true&vCenter=true&repeat=false&width=435&lines=ResNet+Implementation+with+TensorFlow+Keras)](https://git.io/typing-svg)

This repository contains an implementation of the ResNet (Residual Network) architecture using TensorFlow Keras. ResNet is a deep convolutional neural network architecture designed for image classification tasks. It is known for its ability to mitigate the vanishing gradient problem by introducing skip connections, which allow the network to learn identity functions and improve gradient flow.

## ResNet Architecture

ResNet was introduced by Kaiming He et al. in their paper "Deep Residual Learning for Image Recognition" (2015). The key innovation in ResNet is the residual block, which consists of a series of convolutional layers followed by a skip connection that adds the input of the block to its output. This allows the network to learn residual functions, making it easier to train deeper networks.

## Getting Started

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Run the resnet.py script to train and evaluate the ResNet model:
```bash
python resnet.py
```
## Customization
You can customize the ResNet architecture by modifying the ```resnet.py``` script. For example, you can change the number of layers, the number of residual blocks, or the input image size.
