import numpy as np


'''
Initializer


params(shape):
    Create initalized tensor
    @param shape tuple of dimensions of the expected tensor
    @return initialized tensor
    

'''

class GaussInitializer:

    def params(self, shape):
        return np.random.standard_normal(shape)


class GaussSqrtInitializer:

    def params(self, shape):
        mat = np.random.standard_normal(shape)

        if len(shape) > 1:
            mat /= np.sqrt(shape[1])

        return mat
