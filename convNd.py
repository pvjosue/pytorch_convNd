import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Callable
import math

class convNd(nn.Module):
    """Some Information about convNd"""
    def __init__(self,in_channels: int,
                 out_channels: int,
                 num_dims: int,
                 kernel_size: Tuple,
                 stride,
                 padding,
                 is_transposed = False,
                 padding_mode = 'zeros',
                 output_padding = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 rank: int = 0,
                 use_bias: bool = True,
                 bias_initializer: Callable = None,
                 kernel_initializer: Callable = None):
        super(convNd, self).__init__()

        # ---------------------------------------------------------------------
        # Assertions for constructor arguments
        # ---------------------------------------------------------------------
        if not isinstance(kernel_size, Tuple):
            kernel_size = tuple(kernel_size for _ in range(num_dims))
        if not isinstance(stride, Tuple):
            stride = tuple(stride for _ in range(num_dims))
        if not isinstance(padding, Tuple):
            padding = tuple(padding for _ in range(num_dims))
        if not isinstance(output_padding, Tuple):
            output_padding = tuple(output_padding for _ in range(num_dims))
        if not isinstance(dilation, Tuple):
            dilation = tuple(dilation for _ in range(num_dims))

        # This parameter defines which Pytorch convolution to use as a base, for 3 Conv2D is used
        if rank==0 and num_dims<=3:
            max_dims = num_dims-1
        else:
            max_dims = 3

        if is_transposed:
            self.conv_f = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)[max_dims - 1]
        else:
            self.conv_f = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[max_dims - 1]

        assert len(kernel_size) == num_dims, \
            'nD kernel size expected!'
        assert len(stride) == num_dims, \
            'nD stride size expected!'
        assert len(padding) == num_dims, \
            'nD padding size expected!'
        assert len(output_padding) == num_dims, \
            'nD output_padding size expected!'
        assert sum(dilation) == num_dims, \
            'Dilation rate other than 1 not yet implemented!'
        

# ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------
        self.rank = rank
        self.is_transposed = is_transposed
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_dims = num_dims
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer

        # ---------------------------------------------------------------------
        # Construct 3D convolutional layers
        # ---------------------------------------------------------------------
        if self.bias_initializer is not None:
            if self.use_bias:
                self.bias_initializer(self.bias)
        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv_layers = torch.nn.ModuleList()

        # Compute the next dimension, so for a conv4D, get index 3
        next_dim_len = self.kernel_size[0] 
        
        for _ in range(next_dim_len):
            if self.num_dims-1 > max_dims:
                # Initialize a Conv_n-1_D layer
                conv_layer = convNd(in_channels=self.in_channels,
                                            out_channels=self.out_channels,
                                            use_bias=self.use_bias,
                                            num_dims=self.num_dims-1,
                                            rank=self.rank-1,
                                            is_transposed=self.is_transposed,
                                            kernel_size=self.kernel_size[1:],
                                            stride=self.stride[1:],
                                            groups=self.groups,
                                            dilation=self.dilation[1:],
                                            padding=self.padding[1:],
                                            padding_mode=self.padding_mode,
                                            output_padding=self.output_padding[1:],
                                            kernel_initializer=self.kernel_initializer,
                                            bias_initializer=self.bias_initializer)

            else:
                # Initialize a Conv layer
                # bias should only be applied by the top most layer, so we disable bias in the internal convs
                conv_layer = self.conv_f(in_channels=self.in_channels,
                                            out_channels=self.out_channels,
                                            bias=False,
                                            kernel_size=self.kernel_size[1:],
                                            dilation=self.dilation[1:],
                                            stride=self.stride[1:],
                                            padding=self.padding[1:],
                                            padding_mode=self.padding_mode,
                                            groups=self.groups)
                if self.is_transposed:
                    conv_layer.output_padding = self.output_padding[1:]

                # Apply initializer functions to weight and bias tensor
                if self.kernel_initializer is not None:
                    self.kernel_initializer(conv_layer.weight)

            # Store the layer
            self.conv_layers.append(conv_layer)

    # -------------------------------------------------------------------------

    def forward(self, input):
        
        # Pad the input if is not transposed convolution
        if not self.is_transposed:
            padding = list(self.padding)
            # Pad input if this is the parent convolution ie rank=0
            if self.rank==0:
                inputShape = list(input.shape)
                inputShape[2] += 2*self.padding[0]
                padSize = (0,0,self.padding[0],self.padding[0])
                padding[0] = 0
                if self.padding_mode is 'zeros':
                    input = F.pad(input.view(input.shape[0],input.shape[1],input.shape[2],-1),padSize,'constant',0).view(inputShape)
                else:
                    input = F.pad(input.view(input.shape[0],input.shape[1],input.shape[2],-1),padSize,self.padding_mode).view(inputShape)

        # Define shortcut names for dimensions of input and kernel
        (b, c_i) = tuple(input.shape[0:2])
        size_i = tuple(input.shape[2:])
        size_k = self.kernel_size

        if not self.is_transposed:
            # Compute the size of the output tensor based on the zero padding
            size_o = tuple([math.floor((size_i[x] + 2 * padding[x] - size_k[x]) / self.stride[x] + 1) for x in range(len(size_i))])
            # Compute size of the output without stride
            size_ons = tuple([size_i[x] - size_k[x] + 1 for x in range(len(size_i))])
        else:
            # Compute the size of the output tensor based on the zero padding
            size_o = tuple([(size_i[x] - 1) * self.stride[x] - 2 * self.padding[x] + (size_k[x]-1) + 1 + self.output_padding[x] for x in range(len(size_i))])

        # Output tensors for each 3D frame
        frame_results = size_o[0] * [torch.zeros((b,self.out_channels) + size_o[1:], device=input.device)]
        empty_frames = size_o[0] * [None]

        # Convolve each kernel frame i with each input frame j
        for i in range(size_k[0]):
            # iterate inputs first dimmension
            for j in range(size_i[0]):

                # Add results to this output frame
                if self.is_transposed:
                    out_frame = i + j*self.stride[0] - self.padding[0]
                else:
                    out_frame = j - (i - size_k[0] // 2) - (size_i[0] - size_ons[0]) // 2 - (1-size_k[0]%2) 
                    k_center_position = out_frame % self.stride[0]
                    out_frame = math.floor(out_frame / self.stride[0])
                    if k_center_position != 0:
                        continue
                
                if out_frame < 0 or out_frame >= size_o[0]:
                    continue

                # Prepate input for next dimmension
                conv_input = input.view(b, c_i, size_i[0], -1)
                conv_input = conv_input[:, :, j, :].view((b, c_i) + size_i[1:])

                # Convolve
                frame_conv = \
                    self.conv_layers[i](conv_input)

                if empty_frames[out_frame] is None:
                    frame_results[out_frame] = frame_conv
                    empty_frames[out_frame] = 1
                else:
                    frame_results[out_frame] += frame_conv

        result = torch.stack(frame_results, dim=2)

        if self.use_bias:
            resultShape = result.shape
            result = result.view(b,resultShape[1],-1)
            for k in range(self.out_channels):
                result[:,k,:] += self.bias[k]
            return result.view(resultShape)
        else:
            return result
