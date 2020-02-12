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


# These hyperparameters give a 10th epoch validation accuracy of 0.9723
# net = NN(hidden_dims=(700, 200), epsilon=1e-6, lr=2.5e-2, batch_size=64,
#          seed=3491554, activation="relu", data=(train, valid, test), init_method='glorot')
#
# logs = net.train_loop(10)
#
# print(logs['validation_accuracy'][9])

# # Actual H-Param search to populate a table
# for h1 in [600,700]:
#     for h2 in [200,300]:
#         for learn in [1e-2, 2.5e-2]:
#             net = NN(hidden_dims=(h1, h2), epsilon=1e-6, lr=learn, batch_size=64,
#                      seed=3491554, activation="relu", data=(train, valid, test), init_method='glorot')
#
#             logs = net.train_loop(10)
#
#             print("h1:", h1, ", h2:", h2, ", learning rate:", learn, ", 10th epoch validation acc:", logs['validation_accuracy'][9])

# #Testing the finite diff
# net = prob2NN(hidden_dims=(700, 300), epsilon=1e-6, lr=2.5e-2, batch_size=64,
#           seed=3491554, activation="relu", data=(train, valid, test), init_method='glorot')
#
# X_train, y_train = net.train
# y_onehot = y_train
# dims = [X_train.shape[1], y_onehot.shape[1]]
# net.initialize_weights(dims)
#
# print(net.finite_difference(1000))

# #Finite diff for the report question (already trained)
# net = prob2NN(hidden_dims=(700, 300), epsilon=1e-6, lr=2.5e-2, batch_size=64,
#            seed=3491554, activation="relu", data=(train, valid, test), init_method='glorot')
#
# net.train_loop(10)
#
# vals = []
#
# for n in [1,10,50,100,500,1000,10000]:
#     vals.append(net.finite_difference(n))
#
# print(vals)

#Make loss curves for comparison with CNN
net = NN(hidden_dims=(700, 200), epsilon=1e-6, lr=2.5e-2, batch_size=64,
          seed=3491554, activation="relu", data=(train, valid, test), init_method='glorot')

logs = net.train_loop(10)

print(logs['train_loss'])
print(logs['validation_loss'])