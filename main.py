import torch
import torch.nn as nn
import torch.nn.functional as F
from FunctionsNd import ConvNd
import time
import matplotlib.pyplot as plt
from conv4d import Conv4d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
x = torch.randn((4, 1, 50, 50, 10)).to(device)

ks = (5, 5, 5)
padding = 2
stride = 1

conv = ConvNd(1, 1, 3, ks, stride, padding, 
kernel_initializer=lambda x: torch.nn.init.constant_(x, 1), 
bias_initializer=lambda x: torch.nn.init.constant_(x, 0)).to(device)

start = time.time()
out = conv(x)
print("mine")
end = time.time()
print(end - start)

useGT = True

if useGT:
    convGT = nn.Conv3d(1, 1, ks, stride, padding=padding)
    torch.nn.init.constant_(convGT.weight, 1)
    torch.nn.init.constant_(convGT.bias, 0).to(device)

    convGT = convGT.to(device)


    start = time.time()
    outGT = convGT(x)
    print("GT")
    end = time.time()
    print(end - start)
    diff = abs(out-outGT)
    print(torch.sum(diff))

# print(out.shape)
# plt.figure(1)
# plt.subplot(221)
# plt.imshow(x[0,0,:,:,:,:].sum(2).sum(2).cpu().data.detach())
# plt.subplot(222)
# plt.imshow(out[0,0,:,:,:,:].sum(2).sum(2).cpu().data.detach())
# plt.subplot(223)
# plt.imshow(outGT[0,0,:,:,:,:].sum(2).sum(2).cpu().data.detach())
# plt.show()