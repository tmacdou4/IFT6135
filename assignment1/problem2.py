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
