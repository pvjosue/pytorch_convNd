import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Callable
import math

class ConvNd(nn.Module):
    """Some Information about ConvNd"""
    def __init__(self,in_channels: int,
                 out_channels: int,
                 num_dims: int,
                 kernel_size: Tuple,
                 stride,
                 padding,
                 dilation: int = 1,
                 groups: int = 1,
                 use_bias: bool = True,
                 bias_initializer: Callable = None,
                 kernel_initializer: Callable = None):
        super(ConvNd, self).__init__()

        # ---------------------------------------------------------------------
        # Assertions for constructor arguments
        # ---------------------------------------------------------------------
        if not isinstance(kernel_size, Tuple):
            kernel_size = tuple(kernel_size for _ in range(num_dims))
        if not isinstance(stride, Tuple):
            stride = tuple(stride for _ in range(num_dims))
        if not isinstance(padding, Tuple):
            padding = tuple(padding for _ in range(num_dims))

        assert len(kernel_size) == num_dims, \
            '4D kernel size expected!'
        assert len(stride) == num_dims, \
            '4D stride size expected!'
        assert len(padding) == num_dims, \
            '4D padding size expected!'
        assert dilation == 1, \
            'Dilation rate other than 1 not yet implemented!'
        assert groups == 1, \
            'Groups other than 1 not yet implemented!'
        assert num_dims >=3, \
            'This function works for more than 3 dimensions, for less use torch implementation'

# ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_dims = num_dims
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.use_bias = use_bias
        self.weight = torch.nn.Parameter(data=torch.ones(1,num_dims), requires_grad=True)
        self.bias = torch.nn.Parameter(data=torch.ones(1,num_dims), requires_grad=True)
        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer

        # ---------------------------------------------------------------------
        # Construct 3D convolutional layers
        # ---------------------------------------------------------------------
        if self.kernel_initializer is not None:
            self.kernel_initializer(self.weight)
            if self.bias_initializer is not None:
                if self.use_bias:
                    self.bias_initializer(self.bias)
        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv_layers = torch.nn.ModuleList()

        # Compute the next dimension, so for a conv4D, get index 3
        next_dim_len = self.kernel_size[0] 
        
        for _ in range(next_dim_len):
            if self.num_dims-1 != 3:
                # Initialize a Conv_n-1_D layer
                conv_layer = ConvNd(in_channels=self.in_channels,
                                            out_channels=self.out_channels,
                                            use_bias=self.use_bias,
                                            num_dims=self.num_dims-1,
                                            kernel_size=self.kernel_size[1:],
                                            stride=self.stride[1:],
                                            padding=self.padding[1:])

            else:
                # Initialize a Conv3D layer
                conv_layer = torch.nn.Conv3d(in_channels=self.in_channels,
                                            out_channels=self.out_channels,
                                            bias=self.use_bias,
                                            kernel_size=self.kernel_size[1:],
                                            stride=self.stride[1:],
                                            padding=self.padding[1:])

            # Apply initializer functions to weight and bias tensor
            if self.kernel_initializer is not None:
                self.kernel_initializer(conv_layer.weight)
            if self.bias_initializer is not None:
                if self.use_bias:
                    self.bias_initializer(conv_layer.bias)

            # Store the layer
            self.conv_layers.append(conv_layer)

    # -------------------------------------------------------------------------

    def forward(self, input):
        # Define shortcut names for dimensions of input and kernel
        (b, c_i) = tuple(input.shape[0:2])
        size_i = tuple(input.shape[2:])
        size_k = self.kernel_size

        # Compute the size of the output tensor based on the zero padding
        size_o = tuple([math.floor((size_i[x] + 2 * self.padding[x] - size_k[x]) / self.stride[x] + 1) for x in range(len(size_i))])
        # (math.floor((l_i + 2 * self.padding - l_k) / self.stride + 1),

        # Compute size of the output without stride
        size_onp = tuple([size_i[x] + 2 * self.padding[x] - size_k[x] + 1 for x in range(len(size_i))])
        # l_i + 2 * self.padding - l_k + 1

        # Output tensors for each 3D frame
        frame_results = size_o[0] * [torch.zeros((b,c_i) + size_o[1:]).cuda()] # todo: add .cuda()
        empty_frames = size_o[0] * [None]

        # Convolve each kernel frame i with each input frame j
        for i in range(size_k[0]):

            for j in range(size_i[0]):

                # Add results to this output frame
                out_frame = j - (i - size_k[0] // 2) - (size_i[0] - size_onp[0]) // 2 - (1-size_k[0]%2) 
                k_center_position = out_frame % self.stride[0]

                out_frame = math.floor(out_frame / self.stride[0])
                if out_frame < 0 or out_frame >= size_o[0] or k_center_position != 0:
                    continue

                conv_input = input.view(b, c_i, size_i[0], -1)
                conv_input = conv_input[:, :, j, :].view((b, c_i) + size_i[1:])
                # if self.num_dims==4:
                    # conv_input = F.pad(conv_input,(self.padding[0],self.padding[0],self.padding[1],self.padding[1],self.padding[2],self.padding[2]), mode='reflect')
                frame_conv = \
                    self.conv_layers[i](conv_input)

                if empty_frames[out_frame] is None:
                    frame_results[out_frame] = frame_conv
                    empty_frames[out_frame] = 1
                else:
                    frame_results[out_frame] += frame_conv

        return torch.stack(frame_results, dim=2)