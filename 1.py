import torch

x = torch.randn(1, 64, 32, 32)
x.resize_(1, 32, 32, 32)

print(x.shape)