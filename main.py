import torch
import torch.nn as nn
import torch.nn.functional as F
from ndFunctions import ConvNd
import time

x = torch.randn((4, 1, 50, 50, 10))

ks = (2, 2, 2)
padding = 1
stride = 1

conv = ConvNd(1, 1, 3, ks, stride, padding, 
kernel_initializer=lambda x: torch.nn.init.constant_(x, 1), 
bias_initializer=lambda x: torch.nn.init.constant_(x, 0))

convGT = nn.Conv3d(1, 1, ks, stride, padding=padding)
torch.nn.init.constant_(convGT.weight, 1)
torch.nn.init.constant_(convGT.bias, 0)

start = time.time()
out = conv(x)
print("mine")
end = time.time()
print(end - start)
start = time.time()
outGT = convGT(x)
print("GT")
end = time.time()
print(end - start)

diff = abs(out-outGT)
print(torch.sum(diff))
# print(out)
# print(outGT)