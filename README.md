# convNd and convTransposeNd in Pytorch
This n-dimensional convolution is based on recursivly creating a convNd with many conv(N-1)d, until reaching conv3d, where the Pytorch implementation is used. . Also, passing a flag **is_transposed**=True to the convNd function will result in a convTransposeNd operation.

The following convolution features are available:
* bias
* stride
* padding
* output_padding
* groups
Todos:
* dilation

## Examples
The [main.py](https://github.com/pvjosue/pytorch_convNd/blob/master/main.py) contains a convNd test where N=3, in this cased based on multiple conv2d operations. In this way, the functionality of convNd can be compared with the Pytorch conv3d and convTranspose3d operator.
In [mainNd.py](https://github.com/pvjosue/pytorch_convNd/blob/master/mainNd.py), an example with a 5D convolution is presented.

convNd is an extended version of the conv4d from Timothy Gebhard's [code](https://github.com/timothygebhard/pytorch-conv4d).

The convNd was used as a conv4d layer inside the [LFMNet](https://github.com/pvjosue/LFMNet). A CNN capable of reconstructing  3D stacks from Light-field microscopy images.

## Usage
Example in 5D
```python
import torch
from convNd import convNd

# define basic layer info
inChans = 2
outChans = 4
weight = torch.rand(1)[0]
bias = torch.rand(1)[0]

# create input tensor
x = torch.rand(1, inChans, 5, 5, 5, 5, 5).cuda()
conv5d = convNd(
    in_channels=inChans, 
    out_channels=outChans,
    num_dims=5, 
    kernel_size=3, 
    stride=(2,1,1,1,1), 
    padding=0, 
    padding_mode='zeros',
    output_padding=0,
    is_transposed=False,
    use_bias=True, 
    groups=2,
    kernel_initializer=lambda x: torch.nn.init.constant_(x, weight), 
    bias_initializer=lambda x: torch.nn.init.constant_(x, bias)).cuda()


# Create tranposed 5D convolution
# note the is_tranposed parameter
convT5d = convNd(
    in_channels=inChans,
    out_channels=outChans,
    num_dims=5,
    kernel_size=3,
    stride=(2,1,1,1,1),
    padding=0,
    padding_mode='zeros',
    output_padding=0,
    is_transposed=True,
    use_bias=True,
    groups=2,
    kernel_initializer=lambda x: torch.nn.init.constant_(x, weight), 
    bias_initializer=lambda x: torch.nn.init.constant_(x, bias)).cuda()

# apply conv5d
xConv = conv5d(x)
print(xConv.shape)

# apply convTranspose5d
xTConv = convT5d(x)
print(xTConv.shape)
```
