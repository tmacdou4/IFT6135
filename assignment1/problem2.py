from solution import *
import numpy as np

class prob2NN(NN):

    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.weights = {}
        # self.weights is a dictionary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]

        if self.init_method == 'glorot':
            for layer_n in range(1, self.n_hidden + 2):
                temp = np.ones((all_dims[layer_n - 1], all_dims[layer_n])) * (-np.sqrt(6) / np.sqrt(all_dims[layer_n - 1]+all_dims[layer_n]))
                self.weights[f"W{layer_n}"] = temp + (np.random.rand(all_dims[layer_n - 1], all_dims[layer_n]) * (2*np.sqrt(6) / np.sqrt(all_dims[layer_n - 1]+all_dims[layer_n])))
                self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

        elif self.init_method == 'normal':
            for layer_n in range(1, self.n_hidden + 2):
                self.weights[f"W{layer_n}"] = np.random.normal(size=(all_dims[layer_n - 1], all_dims[layer_n]))
                self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

        elif self.init_method == 'zero':
            for layer_n in range(1, self.n_hidden + 2):
                self.weights[f"W{layer_n}"] = np.zeros((all_dims[layer_n - 1], all_dims[layer_n]))
                self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

        else:
            raise Exception("invalid")
        return 0


    #use 5 values from: [k10^i: i \in {1,...,5}, k \in {1,5}
    #call this after training.
    def finite_difference(self, N):
        eps = 1/N

        layer = f"W{2}"

        ex_x = self.train[0][0]
        ex_y = self.train[1][0]

        self.weights[layer] = self.weights[layer] - eps
        for_1 = self.forward(ex_x)

        #add 2 because just subtracted 1 above
        self.weights[layer] = self.weights[layer] + 2*eps
        for_2 = self.forward(ex_x)

        loss_1 = self.loss(for_1[f"Z{self.n_hidden + 1}"], ex_y)
        loss_2 = self.loss(for_2[f"Z{self.n_hidden + 1}"], ex_y)

        grad_pred = (loss_2-loss_1)/(2*eps)

        #reset the weights to what they originally were
        self.weights[layer] = self.weights[layer] - eps
        for_3 = self.forward(ex_x)
        grad_real = self.backward(for_3, ex_y)

        max_diff = np.max(grad_pred[:10] - grad_real[:10])

        return max_diff
