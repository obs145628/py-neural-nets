import dataset_norm4
import test_suite

from activation import SigmoidActivation, SoftmaxActivation, ReluActivation, TanhActivation, LinearActivation
from cost import QuadraticCost, CrossEntropyCost
from initializer import GaussInitializer, GaussSqrtInitializer
from layer import FullyConnectedLayer
from network import Network
import optimizer

X_train, y_train, X_test, y_test = dataset_norm4.load_mini_norm4()

def network1():
    
    layer1 = FullyConnectedLayer(4, 6, activation=SigmoidActivation())
    layer2 = FullyConnectedLayer(6, 10, activation=SigmoidActivation())
    layer3 = FullyConnectedLayer(10, 2, activation=SigmoidActivation())
    layer1.init_weigths(GaussInitializer())
    layer2.init_weigths(GaussInitializer())
    layer3.init_weigths(GaussInitializer())
    return 'sigmoid_quadratic', Network(
        [layer1, layer2, layer3],
        cost_function=QuadraticCost(), test_function= dataset_norm4.output_test
    )


def network2():
    
    layer1 = FullyConnectedLayer(4, 6, activation=SigmoidActivation())
    layer2 = FullyConnectedLayer(6, 10, activation=SigmoidActivation())
    layer3 = FullyConnectedLayer(10, 2, activation=SigmoidActivation())
    layer1.init_weigths(GaussInitializer())
    layer2.init_weigths(GaussInitializer())
    layer3.init_weigths(GaussInitializer())
    return 'sigmoid_cross_entropy', Network(
        [layer1, layer2, layer3],
        cost_function=CrossEntropyCost(), test_function= dataset_norm4.output_test
    )



nets = [network1(), network2()]

ts = test_suite.TestSuite()

ts.begin()
ts.begin_group('backpropagation')

for net in nets:
    ts.begin_test(net[0])
    valid, msg = net[1].check_backpropagation(X_train, y_train)
    ts.end_test(valid, msg)

ts.end_group()
ts.end()
