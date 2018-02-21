import numpy as np

'''
Layer

Layer(input_size, output_size)

input_size: size of the data in input of the layer
output_size: size of the data in input of the layer

forward(x):
@param x (input_size) - input vector
@return vector (output_size)

apply forward propagation on vector x

backpropagation(da):
@param da (output_size) - vector of gradients dC/da
@return vector(input_size)

apply backward propagation to compute dC/d??? for each parameters of the layer
called for each training set, compute from the last call to forward
Cumulate dC/d??? until the call to update wieghts
return vector of gradients dC/dx


update_params(lr, lambda1, lambda2):
@param lr - learning rate
@param lambda1 - coefficient for l1 regularization
@param lambda2 - coefficient for l2 regularization

Update the parameters of the layers according to the cumulated DC/d??? from previous calls to backpropagation


check_grad(net, X, y)
Check the backpropagation algorithm with numerical computations
@param net - neural network
@param X - full training set examples
@param y - full training set labels
@return (difference between backpropagation and numerical computation, details messages)
Assumes backpropagation has already been called on the whole dataset
'''

EPSILON = 1e-8


class FullyConnectedLayer:

    def __init__(self, input_size, output_size, activation):

        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        self.W = np.zeros((output_size, input_size))
        self.b = np.zeros((output_size))

        self.cumul_dW = np.zeros((self.output_size, self.input_size))
        self.cumul_db = np.zeros((self.output_size))

    def get_matrices(self):
        return [self.W, self.b]

    def init_weigths(self, init):
        self.W = init.params((self.output_size, self.input_size))
        self.b = init.params(tuple([self.output_size]))

    def forward(self, x):
        self.x = x
        self.z = np.dot(self.W, self.x) + self.b
        self.a = self.activation.compute(self.z)
        return self.a

    def backpropagation(self, da):
        dz = da * self.activation.compute_prime(self.z)
        dW = np.outer(dz, self.x)
        db = dz
        dx = np.dot(self.W.transpose(), dz)

        self.cumul_dW += dW
        self.cumul_db += db
        return dx

    def update_params(self, lr, lambda1, lambda2):

        new_W = self.W - lr * self.cumul_dW
        new_b = self.b - lr * self.cumul_db

        if lambda1 != 0:
            new_W -= lambda1 * np.sign(self.W)
        if lambda2 != 0:
            new_W -= lambda2 * self.W

        self.W = new_W
        self.b = new_b

        self.cumul_dW = np.zeros((self.output_size, self.input_size))
        self.cumul_db = np.zeros((self.output_size))


    def numerical_backpropagation(self, net, X, y):

        delta_W = np.empty((self.output_size, self.input_size))
        delta_b = np.empty((self.output_size))

        for i in range(0, self.output_size):
            for j in range(0, self.input_size):
                old_val = self.W[i][j]
                self.W[i][j] = old_val + EPSILON
                cost_xp = net.data_cost(X, y)
                self.W[i][j] = old_val - EPSILON
                cost_xn = net.data_cost(X, y)
                delta_W[i][j] = (cost_xp - cost_xn) / (2 * EPSILON)
                self.W[i][j] = old_val

        for i in range(0, self.output_size):
            old_val = self.b[i]
            self.b[i] = old_val + EPSILON
            cost_xp = net.data_cost(X, y)
            self.b[i] = old_val - EPSILON
            cost_xn = net.data_cost(X, y)
            delta_b[i] = (cost_xp - cost_xn) / (2 * EPSILON)
            self.b[i] = old_val

        return delta_W * len(X), delta_b * len(X)


    def check_grad(self, net, X, y):

        num_dW, num_db = self.numerical_backpropagation(net, X, y)
        bac_dW, bac_db = self.cumul_dW, self.cumul_db

        diff_dW = np.linalg.norm(num_dW - bac_dW)
        diff_db = np.linalg.norm(num_db - bac_db)

        msg  = 'dW dist: ' + str(diff_dW) + '\n'
        msg += 'db dist: ' + str(diff_db) + '\n' 
        return max(diff_dW, diff_db), msg
        



class ConvolutionLayer:

    def __init__(self, input_width, input_height, input_depth,
                 init, activation,
                 maps, field_size, stride = 1, padding = 0):

        self.width1 = input_width
        self.height1 = input_height
        self.depth1 = input_depth
        self.init = init
        self.activation = activation
        self.maps_count = maps
        self.field_size = field_size
        self.stride = stride
        self.padding = padding

        self.width2 = (self.width1 - self.field_size + 2 * self.padding) / self.stride + 1
        self.height2 = (self.width2 - self.field_size + 2 * self.padding) / self.stride + 1
        self.depth2 = self.maps_count

        self.W = self.init.params((self.depth2, self.depth1, self.field_size, self.field_size))
        self.b = self.init.params((self.depth2, 1))

    def forward(self, X):

        self.X = X

        self.a = self.activation.compute(self.z)
        return self.a
