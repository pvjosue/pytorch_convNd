# TorchNDFunctions
This repo contains functions for tensors with n-dimensions. 
ConvNd is a nD convolution based on the Timothy Gebhard's [code](https://github.com/timothygebhard/pytorch-conv4d), but extended to support any number of dimensions, this by recursively stacking convolutions from n-dimensions, until getting to conv3d, where the Pytorch implementation is used.

ConvTransposeNd is work in progress. And more functions like BatchNormNd, DropoutNd are coming later.
This convolution operator supports stride and padding, but doesn't support yet dilation and groups.

## Example
  See main.py for an example using Pytorch's conv3d against this ConvNd implementation. 
  The convNd was used as a conv4d for a Light-field microscopy neural network.
