import torch
import torch.nn as nn
from solution import *
import numpy as np

# net = RNN(emb_size=1000, hidden_size=10,
#                 seq_len=10, batch_size=5,
#                 vocab_size=10, num_layers=3,
#                 dp_keep_prob=.5)
#
# #print(net)
#
# i = torch.from_numpy(np.array([1,2,3,4,5]))
# h = torch.rand(3,5,10)
# n = 5
#
# #print(net.generate(i,h,n))
#
# net2 = GRU(emb_size=1000, hidden_size=10,
#                 seq_len=10, batch_size=5,
#                 vocab_size=10, num_layers=3,
#                 dp_keep_prob=0.5)
#
# print(net2.generate(i,h,n))

module = MultiHeadedAttention(5, 25)

q = torch.rand(10,20,25)
k = torch.rand(10,20,25)
v = torch.rand(10,20,25)

print(module.forward(q,k,v))