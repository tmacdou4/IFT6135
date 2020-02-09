import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

import torchvision
import torchvision.transforms

from_numpy = torch.from_numpy

batch_size = 64
num_epochs = 2
store_every = 1000
cuda = torch.cuda.is_available()
if cuda:
    print('cuda is available')
else:
    print('cuda is not available')
