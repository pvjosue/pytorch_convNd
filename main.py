import torch
import torch.nn as nn
import torch.nn.functional as F
from FunctionsNd import ConvNd
from FunctionsNd import ConvTransposeNd
import time
import matplotlib.pyplot as plt
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
inChans = 3
outChans = 91
x = torch.rand((2, inChans, 51, 51, 51)).to(device)

ks = (3, 3, 3)
padding = 1
stride = 4
weight = 2
bias = 0
conv = ConvNd(inChans, outChans, 3, ks, stride, padding, use_bias=True, 
kernel_initializer=lambda x: torch.nn.init.constant_(x, weight), 
bias_initializer=lambda x: torch.nn.init.constant_(x, bias)).to(device)

convT = ConvTransposeNd(1, 1, 3, ks, stride, padding, 
kernel_initializer=lambda x: torch.nn.init.constant_(x, weight), 
bias_initializer=lambda x: torch.nn.init.constant_(x, bias)).to(device)

start = time.time()
out = conv(x)
# outT = convT(x)
print("mine")
end = time.time()
print(end - start)

useGT = True

if useGT:
    convGT = nn.Conv3d(inChans, outChans, ks, stride, padding=padding, bias=True)
    torch.nn.init.constant_(convGT.weight, weight)
    torch.nn.init.constant_(convGT.bias, bias).to(device)

    # convGTT = nn.ConvTranspose3d(1, 1, ks, stride, padding=padding)
    # torch.nn.init.constant_(convGTT.weight, weight)
    # torch.nn.init.constant_(convGTT.bias, bias).to(device)

    convGT = convGT.to(device)
    # convGTT = convGTT.to(device)


    start = time.time()
    outGT = convGT(x)
    # outGTT = convGTT(x)
    print("GT")
    end = time.time()
    print(end - start)
    diff = abs(out-outGT)
    # diffT = abs(outT-outGTT)
    print(torch.sum(diff))
    # print(torch.sum(diffT))

# print(out.shape)
plt.figure(1)
plt.subplot(221)
plt.imshow(x[0,0,:,:,:].sum(2).cpu().data.detach())
plt.title('input')
plt.subplot(222)
plt.imshow(out[0,:,:,:,:].sum(2).sum(1).cpu().data.detach())
plt.title('convND out')
plt.subplot(223)
plt.imshow(outGT[0,:,:,:,:].sum(2).sum(1).cpu().data.detach())
plt.title('GT out')
plt.subplot(224)
plt.imshow(diff[0,:,:,:,:].sum(2).sum(1).cpu().data.detach())
plt.title('diff out')
plt.show()