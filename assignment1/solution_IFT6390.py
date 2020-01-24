import pickle
import numpy as np

#modified from solution.py template

class NN(object):
    def __init__(self,
                 hidden_dims=(512, 256),
                 datapath='cifar10.pkl',
                 n_classes=10,
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=1000,
                 seed=None,
                 activation="relu",
                 init_method="glorot"
                 ):
        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if datapath is not None:
            u = pickle._Unpickler(open(datapath, 'rb'))
            u.encoding = 'latin1'
            self.train, self.valid, self.test = u.load()
        else:
            self.train, self.valid, self.test = None, None, None

    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            #must be initialized randomly over the range (-1/sqrt(neuron_input_dimensions)) to (+1/sqrt(neuron_input_dimensions))
            #Set to bottom of range
            temp = np.ones((all_dims[layer_n - 1], all_dims[layer_n])) * (-1 / np.sqrt(all_dims[layer_n - 1]))
            #add rand in (0,1) * width of range (2/sqrt(neuron_input_dimensions))
            self.weights[f"W{layer_n}"] = temp + (np.random.rand(all_dims[layer_n - 1], all_dims[layer_n]) * (2 / np.sqrt(all_dims[layer_n - 1])))
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

    def print_weights(self, size=False):
        if size:
            for i in range(1, self.n_hidden + 2):
                print("Weight", i, "shape:", self.weights[f"W{i}"].shape)
                print("Bias", i, "shape:", self.weights[f"b{i}"].shape)
        else:
            print(self.weights)

    def relu(self, x, grad=False):
        if grad:
            return 1 * (x > 0)
        return x * (x > 0)

    def sigmoid(self, x, grad=False):
        #Numerically stable sigmoid function
        temp = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        if grad:
            return temp*(1-temp)
        return temp

    def tanh(self, x, grad=False):
        #using a couple temp variable to avoid calculating stuff twice.
        temp_1 = np.exp(x)
        temp_2 = np.exp(-x)
        temp_3 = (temp_1-temp_2)/(temp_1+temp_2)
        if grad:
            return 1 - temp_3 ** 2
        return temp_3

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            return self.relu(x, grad=grad)
        elif self.activation_str == "sigmoid":
            return self.sigmoid(x, grad=grad)
        elif self.activation_str == "tanh":
            return self.tanh(x, grad=grad)
        else:
            raise Exception("invalid")
        return 0

    def softmax(self, x):
        # Remember that softmax(x-C) = softmax(x) when C is a constant.

        #Check the dimensionality of the input because the autograder wants a (n, ) size matrix when x is
        #1D but a (n,m) size matrix when x is 2d
        if x.ndim == 1:
            max_val = np.max(x)
            x = x - max_val
            e_x = np.exp(x)
            output = e_x / np.sum(e_x)
            return output
        # Stabilize by removing the max
        max_val = np.max(x, axis=1)
        x = x - max_val[:, np.newaxis]
        e_x = np.exp(x)
        output = e_x / np.sum(e_x, axis=1)[:, np.newaxis]
        return output

    def forward(self, x):
        cache = {"Z0": x}
        # cache is a dictionary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
        for layer_n in range(1, self.n_hidden+2):
            cache[f"A{layer_n}"] = self.weights[f"b{layer_n}"] + np.matmul(cache[f"Z{layer_n-1}"], self.weights[f"W{layer_n}"])
            if layer_n == self.n_hidden+1:
                cache[f"Z{layer_n}"] = self.softmax(cache[f"A{layer_n}"])
            else:
                cache[f"Z{layer_n}"] = self.activation(cache[f"A{layer_n}"])
        return cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}
        # grads is a dictionary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
        grads[f"dA{self.n_hidden+1}"] = output-labels
        for layer_n in range(self.n_hidden+1, 0, -1):
            grads[f"dW{layer_n}"] = np.matmul(cache[f"Z{layer_n-1}"].T, grads[f"dA{layer_n}"])/cache[f"Z{layer_n-1}"].shape[0]
            grads[f"db{layer_n}"] = np.sum(grads[f"dA{layer_n}"], axis=0, keepdims=True)/grads[f"dA{layer_n}"].shape[0]
            if layer_n > 1:
                grads[f"dZ{layer_n-1}"] = np.matmul(self.weights[f"W{layer_n}"], grads[f"dA{layer_n}"].T).T
                grads[f"dA{layer_n-1}"] = grads[f"dZ{layer_n-1}"] * self.activation(cache[f"A{layer_n-1}"], grad=True)
        return grads

    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            self.weights[f"W{layer}"] -= self.lr * grads[f"dW{layer}"]
            self.weights[f"b{layer}"] -= self.lr * grads[f"db{layer}"]

    def one_hot(self, y):
        output = np.zeros((len(y), self.n_classes))
        for i in range(len(y)):
            output[i, y[i]] = 1
        return output

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        entropy_vals = -(labels * np.log(prediction))
        loss = np.sum(entropy_vals)/labels.shape[0]
        return loss

    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        y_onehot = self.one_hot(y_train)
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                cache = self.forward(minibatchX)
                grads = self.backward(cache, minibatchY)
                self.update(grads)

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
        return test_loss, test_accuracy
