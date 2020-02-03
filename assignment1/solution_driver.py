from solution import *
from problem2 import *
import numpy as np

#Tester for solution.py

#Testing init method switching

net = prob2NN(activation="relu", hidden_dims=(300, 50), seed = 42069, init_method='glorot')
net.initialize_weights((784, 10))
print(np.mean(net.weights[f"W{1}"]))
print(np.mean(net.weights[f"W{2}"]))
print(np.mean(net.weights[f"W{3}"]))

print(np.max(net.weights[f"W{1}"]))
print(np.max(net.weights[f"W{2}"]))
print(np.max(net.weights[f"W{3}"]))
