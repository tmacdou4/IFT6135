from q1_solution import *
import numpy as np
import torch

# x = torch.rand(1,1)-0.5
# y = torch.randint(0,2, size=(1, 1))
# z = torch.rand(1,1)*0.1
#
# print(x)
# print(y)
# print(z)
#
# print(log_likelihood_normal(x, z, y))
torch.manual_seed(5)
a = torch.Tensor([[10,2,300],[15,300,5]])
b = torch.Tensor([[5,0.2,5],[5,0.2,5]])

c = torch.Tensor([[9,2.5,295],[14,302,5.5]])
d = torch.Tensor([[5,0.2,5],[5,0.2,5]])

print(kl_gaussian_gaussian_mc(a, b, c, d))