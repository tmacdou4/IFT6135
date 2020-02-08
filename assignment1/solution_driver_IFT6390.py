from solution_IFT6390 import *
import numpy as np

#driver to test solution.py

#Testing weights (Question 1)
# net = NN(activation='relu')
# net.initialize_weights((4, 3))
# net.print_weights(size=True)

#Testing activation functions (Question 2)
# net = NN(activation='relu')
# x = -.5
# y = np.array([-1,0.1,-.25,.3, 1])
# z = np.array([[-1,0.1,-.25,.3, 1],[1,-0.1,.25,.3,-1]])
# # # # print(net.relu(x))
# # # print(net.relu(y))
# # # print(net.relu(x, grad=True))
# # # print(net.relu(y, grad=True))
# #
# print(net.sigmoid(x))
# print(net.sigmoid(z))
# print(net.sigmoid(x, grad=True))
# print(net.sigmoid(y, grad=True))
#
# # print(net.tanh(x))
# # print(net.tanh(y))
# # print(net.tanh(x, grad=True))
# # print(net.tanh(y, grad=True))
#
# print(net.activation(y))
# print(net.activation(z))
# print(net.activation(y, grad=True))
# print(net.activation(z, grad=True))

#Testing question 3
# net = NN()
# z = np.array([-1,0.1,-.25,.3, 1])
# w = np.array([[0,-0.5,-.25,-.3, -1], [1,0.5,0.75,.7, 0]])
#
# print(net.softmax(w))

#Testing question 4
# x = np.array([[-1,0.1,-.25,.3, 1]])
# w = np.array([[0,-0.5,-.25,-.3, -1], [1,0.5,0.75,.7, 0]])
#
# net = NN(activation='sigmoid', hidden_dims=([3]), seed=20136425)
# net.initialize_weights((5, 2))
# print(net.forward(w))

#Testing question 5, requires using question4
# x = np.array([[-1,0.1,-.25,.3, 1]])
# w = np.array([[-1,0.1,-.25,.3, 1], [1,0.5,0.75,.7, 0]])
# n_input = 5
# n_output = 3
#
# net = NN(activation='sigmoid', hidden_dims=([4]), seed=20136425)
# net.initialize_weights((n_input, n_output))
# cache = net.forward(w)
# labels = np.array([[1,0,0],[0,1,0]])
# print(cache)
# print(net.backward(cache, labels))

#Questions 6, gradient decsent update.
# w = np.array([[-1,0.1,-.25,.3, 1], [1,0.5,0.75,.7, 0]])
# n_input = 5
# n_output = 3
#
# net = NN(activation='sigmoid', hidden_dims=([4]), seed=20136425)
# net.initialize_weights((n_input, n_output))
# cache = net.forward(w)
# labels = np.array([[1,0,0],[0,1,0]])
# grads = net.backward(cache, labels)
# net.print_weights()
# net.update(grads)
# net.print_weights()

# Testing question 7
# net = NN(n_classes=4)
# x = np.array([2,0,3])
#
# y = np.array([[.001,.001,.997,.001],[.997,.001,.001,.001],[.001,.001,.001,.997]])
# y = np.array([[.01,.01,.97,.001],[.97,.01,.01,.01],[.01,.01,.01,.97]])
# y = np.array([[.2,.2,.2,.4],[.1,.1,.5,.3],[.1,.4,.4,.1]])
# l = net.one_hot(x)
# print(net.loss(y, l))

# Test to see if the loss goes down after a weight update
# w = np.array([[-1,0.1,-.25,.3, 1], [1,0.5,0.75,.7, 0]])
# n_input = 5
# n_output = 3
#
# net = NN(activation='sigmoid', hidden_dims=([4]), seed=20136425, lr=.5)
# net.initialize_weights((n_input, n_output))
# cache = net.forward(w)
# labels = np.array([[1,0,0],[0,1,0]])
# print(net.loss(cache[f"Z2"], labels))
# grads = net.backward(cache, labels)
# net.update(grads)
# cache = net.forward(w)
# print(net.loss(cache[f"Z2"], labels))

# Testing question 8. First test that needs actual data
# net = NN(activation='sigmoid', seed=20136425, datapath="cifar10.pkl")
# print(net.train_loop(2))


