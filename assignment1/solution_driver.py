from solution import *
import numpy as np

#Tester for solution.py

#Testing weights

net = NN(activation="relu", hidden_dims=(300, 50))
net.initialize_weights((784, 10))
print(np.mean(net.weights[f"W{1}"]))
print(np.mean(net.weights[f"W{2}"]))
print(np.mean(net.weights[f"W{3}"]))

print(np.max(net.weights[f"W{1}"]))
print(np.max(net.weights[f"W{2}"]))
print(np.max(net.weights[f"W{3}"]))

#Testing activations