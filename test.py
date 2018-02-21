import dataset_mnist

from activation import SigmoidActivation, SoftmaxActivation, ReluActivation, TanhActivation, LinearActivation
from cost import QuadraticCost, CrossEntropyCost
from initializer import GaussInitializer, GaussSqrtInitializer
from layer import FullyConnectedLayer
from network import Network
import optimizer


X_train, y_train, X_test, y_test = dataset_mnist.load_mnist()

layer1 = FullyConnectedLayer(784, 100, activation=SigmoidActivation())
layer2 = FullyConnectedLayer(100, 10, activation=SigmoidActivation())
layer1.init_weigths(GaussInitializer())
layer2.init_weigths(GaussInitializer())
net = Network(
    [layer1, layer2],
    cost_function=CrossEntropyCost(), test_function= dataset_mnist.output_test
)
opti = optimizer.SGDOptimizer(lr = 0.5, batch_size = 10, lambda1=0.0, lambda2=5.0)



net.evaluate(X_test, y_test)
net.train(X_train, y_train, X_test, y_test, opti, 30,
          disp_train_cost=True)

