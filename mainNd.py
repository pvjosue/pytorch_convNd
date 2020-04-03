import torch
import torch.nn as nn
import torch.nn.functional as F
from convNd import convNd
import time
import matplotlib.pyplot as plt
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define 4D tensor to test on
inChans = 1
outChans = 1
x = torch.ones(1, inChans, 5, 5, 5, 5, 5).to(device)
ks = 3
padding = 0
# Only zeros allowed by pytorch
padding_mode = 'zeros'
stride = 2
weight = 1
bias = 0
groups = 1

# ConvNd where d = 5
conv = convNd(inChans, outChans, 5, ks, stride, padding, use_bias=True, 
padding_mode=padding_mode, groups=groups,
kernel_initializer=lambda x: torch.nn.init.constant_(x, weight), 
bias_initializer=lambda x: torch.nn.init.constant_(x, bias)).to(device)

# Transposed convolution
convT = convNd(inChans, outChans, 5, ks, stride, padding, 
groups=groups, is_transposed=True,
kernel_initializer=lambda x: torch.nn.init.constant_(x, weight), 
bias_initializer=lambda x: torch.nn.init.constant_(x, bias)).to(device)

# Run timers to compare with torch implementations
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
print(' ')

# Convolve with ConvNd
torch.cuda.synchronize()
start.record()
out = conv(x)
end.record()
torch.cuda.synchronize()
print("ConvNd time: " + str(start.elapsed_time(end)))
print(out.shape)
# Convolve with ConvTransposeNd
torch.cuda.synchronize()
start.record()
outT = convT(x)
end.record()
torch.cuda.synchronize()
print("ConvTransposeNd time: " + str(start.elapsed_time(end)))
print(outT.shape)
