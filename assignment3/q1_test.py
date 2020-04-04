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
y = torch.rand(3,4)

print(log_mean_exp(y))