import numpy as np

class Network:

    '''
    @param layers array of Layers objects
    @param cost_function CostFunction object, represents the function to minimize
    @param test_function predicate taking y_hat and y (output data and expected data)
    '''
    def __init__(self, layers, cost_function, test_function):
        self.layers = layers
        self.cost = cost_function
        self.test = test_function

    '''
    Apply a feed forward pass into all layers
    Only take one example at a time
    @param X - a row vector
    @return y_hat - row vector  
    '''
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    '''
    Apply back propagation algorithm in one input to compute dC/d?? (gradients of params for each layer)
    @param x vector of input data
    @param y vector of expected output data
    '''
    def backpropagation(self, x, y):
        y_hat = self.forward(x)
        da = self.cost.cost_derivative(y, y_hat)
        for l in range(len(self.layers) - 1, -1, -1):
            da = self.layers[l].backpropagation(da)

    '''
    Check backpropagation on ech layer of the neural network
    @param X - matrix of the whole input data
    @param y - matrix of the whole output data
    @return (true if all layers are correct, false otherwhise, details message)
    '''
    def check_backpropagation(self, X, y):
        valid = True
        for i in range(len(X)):
            self.backpropagation(X[i], y[i])

        msg = ''

        for i in range(len(self.layers)):
            msg += 'Layer ' + str(i + 1) + ':\n'
            val = self.layers[i].check_grad(self, X, y)
            msg += val[1]
            succ = val[0] < 1e-2
            if not succ:
                valid = False

        return valid, msg


    '''
    Evaluate data set and print results
    @param X - matrix of input data
    @param y - matrix of expected output data
    '''
    def evaluate(self, X, y):

        n = len(X)
        valid = 0

        for i in range(0, n):
            if self.test(self.forward(X[i]), y[i]):
                valid = valid + 1

        acc = float(valid) / n * 100.0
        print("Evaluation: " + str(valid) + "/" + str(n) + " (" + str(acc) + "%)")


    '''
    Compute the cost error function for a whole data set
    @param X input data
    @param y expected output data
    @return cost error value
    '''
    def data_cost(self, X, y):

        n = len(X)

        res = 0

        for i in range(0, n):
            X_ex = X[i]
            y_ex = y[i]
            y_hat = self.forward(X_ex)
            res += self.cost.cost(y_ex, y_hat)

        res /= n
        return res

    '''
    Train and evaluate the network for n epochs
    @param X_train input train data
    @param y_train expected output train data
    @pram X_test input test data
    @param y_test expected output test_data
    @param opti - optimizer
    @param epochs number of epochs of training
    @param dist_train_cost if true, display the cost error value on the training set
    '''
    def train(self, X_train, y_train, X_test, y_test, opti, epochs,
              disp_train_cost = False):
        for i in range(1, epochs + 1):
            print("Epoch " + str(i) + ":")
            opti.run(self, X_train, y_train)
            self.evaluate(X_test, y_test)

            if disp_train_cost:
                print("Train Cost: " + str(self.data_cost(X_train, y_train)))
        
