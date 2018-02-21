import numpy as np

def shuffle2(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]


'''
Optimizer():

Optimize the parameters of a specific neural network

run(net, X, y):
@param net - neural network to be optimized
@param X (set_len, input_size) - input train data
@param y (set_len, output_size) - expected output train data

run one epoch of optimization and update the parameters of the network


cost(net, X, y):
@param net - neural network
@param X (set_len, input_size) - input data
@param y (set_len, output_size) - expected output data
@return cost value

Return the result of value of the cost function
The cost function of the network is not used if the optimizer compute it differently (regularization)
'''


class SGDOptimizer:

    '''
    @param lr - learning rate
    @param batch_size - size of a mini batch
    @param lambda1 - coefficient for L1 regularization
    @param lambda2 - coefficient for L2 regularization
    '''
    def __init__(self, lr, batch_size, lambda1 = 0.0, lambda2 = 0.0):
        self.lr = lr
        self.batch_size = batch_size
        self.lambda1 = lambda1
        self.lambda2 = lambda2



    '''
    @param net - neural network to be optimized
    @param X - matrix n * p containing training input
    @param y - matrix n * q contraining training expected results
    
    stochastic Gradient Descent:
    Shuffle data
    Divide data in mini_batches of len batch_size and apply stochastic gradient descent on each of them
    for each mini batch:
      for each training example x run backpropagation
      for each layer update params using learning rate, l1 regularization and l2 regularization
    '''
    def run(self, net, X, y):

        X, y = shuffle2(X, y)
        n = len(X)

        for k in range(0, n, self.batch_size):
            X_batch = X[k:k + self.batch_size]
            y_batch = y[k:k + self.batch_size]
            m = len(X_batch)

            for i in range(0, m):
                net.backpropagation(X_batch[i], y_batch[i])
            for l in net.layers:
                l.update_params(self.lr / m, self.lr * self.lambda1 / n, self.lr * self.lambda2 / n)

        #for l in self.layers: debug_tensors.debug_add(l.W, l.b)
        

    def cost(self, net, X, y):
        return net.data_cost(X, y, self.lambda1, self.lambda2)
