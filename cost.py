import numpy as np

'''
Cost

cost(y, y_hat):
Compute the cost error value for one input vector
@param y expected output vector
@param y_hat network output vector
@return cost error value

cost_derivative(y, y_hat):
Compute the cost error derivative at output layer dC/da
@param y expected output vector
@param y_hat network output vector
@return vector dC/da
'''

'''
Quadratic cost
Also called Mean Squared Error (MSE)
or error function
'''
class QuadraticCost:

    def __init__(self):
        self.name = 'quadratic'

    def cost(self, y, y_hat):
        return 0.5 * np.dot(y - y_hat, y - y_hat)

    def cost_derivative(self, y, y_hat):
        return y_hat - y

'''
Cross Entrop Cost
'''
class CrossEntropyCost:

    def __init__(self):
        self.name = 'cross_entropy'

    def cost(self, y, y_hat):
        return np.sum(np.nan_to_num(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)))

    def cost_derivative(self, y, y_hat):
        div = y_hat * (1 - y_hat)
        return (y_hat - y) / div



