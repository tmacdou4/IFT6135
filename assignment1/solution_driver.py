from solution import *
from problem2 import *
import numpy as np

#Tester for solution.py

#Testing init method switching

# net = prob2NN(activation="relu", hidden_dims=(300, 50), seed = 42069, init_method='glorot')
# net.initialize_weights((784, 10))
# print(np.mean(net.weights[f"W{1}"]))
# print(np.mean(net.weights[f"W{2}"]))
# print(np.mean(net.weights[f"W{3}"]))
#
# print(np.max(net.weights[f"W{1}"]))
# print(np.max(net.weights[f"W{2}"]))
# print(np.max(net.weights[f"W{3}"]))


#probably the way data was meant to be loaded in this.
train, valid, test = load_mnist()

#for report problem 2, initialization
# for init in ("glorot", "normal", "zero"):
#     net = prob2NN(hidden_dims=(550, 225), epsilon=1e-6, lr=7e-4, batch_size=64,
#         seed=3491554, activation="relu", data=(train, valid, test), init_method=init)
#
#
#     logs = net.train_loop(10)
#
#     print(logs['train_loss'])


#These hyperparameters give a 10th epoch validation accuracy of 0.9715
# net = NN(hidden_dims=(700, 300), epsilon=1e-6, lr=2.5e-2, batch_size=64,
#          seed=3491554, activation="relu", data=(train, valid, test), init_method='glorot')
#
# logs = net.train_loop(10)
#
# print(logs['validation_accuracy'][9])