import os
os.chdir('D:\\NeuralNetwork\\Network1')

import loader
train, validate, test = loader.load_data_wrapper()

import network

net = network.Network([784, 30, 10])

net.SGD(train, 30, 10, 3.0, test_data=test)

