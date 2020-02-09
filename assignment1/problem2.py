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

        p = min(10, len(self.weights[layer]))
        max_diff = 0

        for i in range(p):

            self.weights[layer][i] = self.weights[layer][i] - eps
            for_1 = self.forward(ex_x)

            #add 2 because just subtracted 1 above
            self.weights[layer][i] = self.weights[layer][i] + 2*eps
            for_2 = self.forward(ex_x)

            loss_1 = self.loss(for_1[f"Z{self.n_hidden + 1}"], ex_y)
            loss_2 = self.loss(for_2[f"Z{self.n_hidden + 1}"], ex_y)

            grad_pred = (loss_2-loss_1)/(2*eps)

            #reset the weights to what they originally were
            self.weights[layer][i] = self.weights[layer][i] - eps
            for_3 = self.forward(ex_x)
            grad_real_dict = self.backward(for_3, ex_y)
            grad_real = grad_real_dict[f"dW{2}"]

            diff = np.max(grad_pred - grad_real[i])

            if np.abs(diff) > np.abs(max_diff):
                max_diff = diff

        return max_diff
