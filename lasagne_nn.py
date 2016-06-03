#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import pickle
import lasagne.layers.dnn
import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX

import ipdb


# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_mlp(input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out


def build_custom_mlp(input_var=None, depth=2, width=800, drop_input=.2,
                     drop_hidden=.5):
    # By default, this creates the same network as `build_mlp`, but it can be
    # customized with respect to the number and size of hidden layers. This
    # mostly showcases how creating a network in Python code can be a lot more
    # flexible than a configuration file. Note that to make the code easier,
    # all the layers are just called `network` -- there is no need to give them
    # different names if all we return is the last one we created anyway; we
    # just used different names above for clarity.

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(
                network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax)
    return network

def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 150, 150),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = ConvLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = ConvLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=9,
            nonlinearity=lasagne.nonlinearities.linear)

    return network


def build_cnn_pose_embedding(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 150, 150),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = ConvLayer(
            network, num_filters=128, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=256, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=256, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=256, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    # network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=128,
            nonlinearity=lasagne.nonlinearities.rectify)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
        network,
            num_units=32,
            nonlinearity=lasagne.nonlinearities.linear)

    return network

def build_cnn_small(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 150, 150),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = ConvLayer(
            network, num_filters=16, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = ConvLayer(
            network, num_filters=16, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=64,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=9,
            nonlinearity=lasagne.nonlinearities.linear)

    return network

def build_cnn2(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 150, 150),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = ConvLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = ConvLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = ConvLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=9,
            nonlinearity=lasagne.nonlinearities.linear)

    return network

def build_cnn_pose(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 150, 150),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = ConvLayer(
            network, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=128, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=32,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            network,
            num_units=4,
            nonlinearity=lasagne.nonlinearities.linear)

    return network


def build_cnn_pose_color(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 3, 150, 150),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = ConvLayer(
            network, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=128, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=32,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            network,
            num_units=4,
            nonlinearity=lasagne.nonlinearities.linear)

    return network

def build_cnn_appLight(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    input = lasagne.layers.InputLayer(shape=(None, 3, 150, 150),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = ConvLayer(
            input, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=128, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=32,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            network,
            num_units=12,
            nonlinearity=lasagne.nonlinearities.linear)

    return network

def build_cnn_shape(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 150, 150),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = ConvLayer(
            network, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=128, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=32,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            network,
            num_units=10,
            nonlinearity=lasagne.nonlinearities.linear)

    return network

def build_cnn_shape_k(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    input = lasagne.layers.InputLayer(shape=(None, 3, 150, 150),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = ConvLayer(
            input, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network =  ConvLayer(
            network, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=128, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network =  ConvLayer(
            network, num_filters=128, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.0),
            num_units=128,
            nonlinearity=lasagne.nonlinearities.rectify)

    ## A fully-connected layer of 256 units with 50% dropout on its inputs:
    #network = lasagne.layers.DenseLayer(
    #        lasagne.layers.dropout(network, p=.0),
    #        num_units=128,
    #        nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            network,
            num_units=10,
            nonlinearity=lasagne.nonlinearities.linear)

    return network

def build_cnn_shape_medium(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    input = lasagne.layers.InputLayer(shape=(None, 3, 150, 150),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = ConvLayer(
            input, num_filters=128, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=128, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=128, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=128,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            network,
            num_units=10,
            nonlinearity=lasagne.nonlinearities.linear)

    return network

def build_cnn_shape_large(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 3, 150, 150),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = ConvLayer(
            network, num_filters=256, filter_size=(7, 7),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = ConvLayer(
            network, num_filters=256, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = ConvLayer(
            network, num_filters=256, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = ConvLayer(
            network, num_filters=256, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            network,
            num_units=10,
            nonlinearity=lasagne.nonlinearities.linear)

    return network


def build_cnn_light(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    input = lasagne.layers.InputLayer(shape=(None, 3, 150, 150),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = ConvLayer(
            input, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=128, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=32,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            network,
            num_units=9,
            nonlinearity=lasagne.nonlinearities.linear)

    return network

def build_cnn_app(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    input = lasagne.layers.InputLayer(shape=(None, 3, 150, 150),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = ConvLayer(
            input, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=128, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=32,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            network,
            num_units=3,
            nonlinearity=lasagne.nonlinearities.linear)


    return network

class MeanLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return input.mean(axis=1).mean(axis=1)

    def get_output_shape_for(self, input_shape):
        return [input_shape[0],1]

def build_cnn_mask(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    input = lasagne.layers.InputLayer(shape=(None, 3, 150, 150),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = ConvLayer(
            input, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=128, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=225,
            nonlinearity=lasagne.nonlinearities.rectify)


    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=2250,
            nonlinearity=lasagne.nonlinearities.linear)

    numSmallMask = 2250
    scaleFactor = 10
    network = lasagne.layers.ReshapeLayer(network, ([0],1, numSmallMask))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    mask = lasagne.layers.Upscale1DLayer(
            incoming=network,
            scale_factor=scaleFactor)

    mask = lasagne.layers.ReshapeLayer(mask, ([0], numSmallMask*scaleFactor))

    mask = lasagne.layers.NonlinearityLayer(
            mask,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    return mask

def build_cnn_mask_large(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    input = lasagne.layers.InputLayer(shape=(None, 3, 150, 150),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = ConvLayer(
            input, num_filters=256, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    lasagne.layers.get_output_shape(network)
    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=256, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=256, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network =  ConvLayer(
            network, num_filters=256, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    # network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1, 1))

    network =  ConvLayer(
            network, num_filters=256, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    # network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=1125,
            nonlinearity=lasagne.nonlinearities.rectify)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=1125,
            nonlinearity=lasagne.nonlinearities.rectify)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network),
            num_units=2500,
            nonlinearity=lasagne.nonlinearities.linear)

    # numSmallMask = 2250
    # scaleFactor = 10
    # network = lasagne.layers.ReshapeLayer(network, ([0],1, numSmallMask))
    #
    # # A fully-connected layer of 256 units with 50% dropout on its inputs:
    # mask = lasagne.layers.Upscale1DLayer(
    #         incoming=network,
    #         scale_factor=scaleFactor)
    #
    # mask = lasagne.layers.ReshapeLayer(mask, ([0], numSmallMask*scaleFactor))

    mask = lasagne.layers.NonlinearityLayer(
            network,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    return mask

def build_cnn_appmask(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    input = lasagne.layers.InputLayer(shape=(None, 3, 150, 150),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = ConvLayer(
            input, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network =  ConvLayer(
            network, num_filters=128, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=2250,
            nonlinearity=lasagne.nonlinearities.rectify)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    mask = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=22500,
            nonlinearity=lasagne.nonlinearities.rectify)


    inputR = lasagne.layers.SliceLayer(input, 0, axis=1)
    inputG = lasagne.layers.SliceLayer(input, 1, axis=1)
    inputB = lasagne.layers.SliceLayer(input, 2, axis=1)


    reshapedMask = lasagne.layers.ReshapeLayer(mask, ([0],150, 150))

    outR = MeanLayer(lasagne.layers.ElemwiseMergeLayer(incomings=[inputR, reshapedMask], merge_function=theano.tensor.mul))
    outG = MeanLayer(lasagne.layers.ElemwiseMergeLayer(incomings=[inputG, reshapedMask], merge_function=theano.tensor.mul))
    outB = MeanLayer(lasagne.layers.ElemwiseMergeLayer(incomings=[inputB, reshapedMask], merge_function=theano.tensor.mul))

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:

    network = lasagne.layers.ConcatLayer(incomings=[outR, outG, outB], axis=0)

    network = lasagne.layers.ReshapeLayer(network, ([0],3))

    return network

def build_cnn_pose_norm(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 150, 150),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.normalization.batch_norm(ConvLayer(
            network, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify))

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.normalization.batch_norm(ConvLayer(
            network, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify))

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.normalization.batch_norm(ConvLayer(
            network, num_filters=128, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify))

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.normalization.batch_norm(lasagne.layers.DenseLayer(
            network,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.normalization.batch_norm(lasagne.layers.DenseLayer(
            network,
            num_units=32,
            nonlinearity=lasagne.nonlinearities.rectify))

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            network,
            num_units=4,
            nonlinearity=lasagne.nonlinearities.linear)


    return network

def build_cnn_pose_large(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 150, 150),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = ConvLayer(
            network, num_filters=96, filter_size=(7, 7),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = ConvLayer(
            network, num_filters=256, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = ConvLayer(
            network, num_filters=512, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = ConvLayer(
            network, num_filters=512, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = ConvLayer(
            network, num_filters=512, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=4096,
            nonlinearity=lasagne.nonlinearities.rectify)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=1024,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            network,
            num_units=4,
            nonlinearity=lasagne.nonlinearities.linear)

    return network

def build_cnn_pose_large_norm(input_var=None):

    network = lasagne.layers.InputLayer(shape=(None, 1, 150, 150),
                                        input_var=input_var)

    network = lasagne.layers.normalization.batch_norm(ConvLayer(
            network, num_filters=96, filter_size=(7, 7),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform()))

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3))

    network = lasagne.layers.normalization.batch_norm(ConvLayer(
            network, num_filters=256, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify))

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.normalization.batch_norm(ConvLayer(
            network, num_filters=512, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify))

    network = lasagne.layers.normalization.batch_norm(ConvLayer(
            network, num_filters=512, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify))

    network = lasagne.layers.normalization.batch_norm(ConvLayer(
            network, num_filters=512, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify))
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3))

    network = lasagne.layers.normalization.batch_norm(lasagne.layers.DenseLayer(
            network,
            num_units=4096,
            nonlinearity=lasagne.nonlinearities.rectify))

    network = lasagne.layers.normalization.batch_norm(lasagne.layers.DenseLayer(
            network,
            num_units=1024,
            nonlinearity=lasagne.nonlinearities.rectify))

    network = lasagne.layers.DenseLayer(
            network,
            num_units=4,
            nonlinearity=lasagne.nonlinearities.linear)

    return network

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def iterate_minibatches_h5(inputs_h5, trainSet, trainValSet, targets, batchsize, shuffle=False):
    assert len(trainValSet) == len(targets)
    print("Loading minibatch set")
    if shuffle:
        indices = np.arange(len(trainValSet))
        np.random.shuffle(indices)
    for start_idx in range(0, len(trainValSet), batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        boolSet = np.zeros(len(inputs_h5)).astype(np.bool)
        boolSet[trainSet[trainValSet[excerpt]]] = True
        yield inputs_h5[boolSet,:,:], targets[excerpt]
    print("Ended loading minibatch set")

def load_network(modelType='cnn', param_values=[]):
    # Load the dataset

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if modelType == 'mlp':
        network = build_mlp(input_var)
    elif modelType.startswith('custom_mlp:'):
        depth, width, drop_in, drop_hid = modelType.split(':', 1)[1].split(',')
        network = build_custom_mlp(input_var, int(depth), int(width),
                                   float(drop_in), float(drop_hid))
    elif modelType == 'cnn':
        network = build_cnn(input_var)
    elif modelType == 'cnn2':
        network = build_cnn2(input_var)
    elif modelType == 'cnn_pose':
        network = build_cnn_pose(input_var)
    elif modelType == 'cnn_pose_large':
        network = build_cnn_pose_large(input_var)
    elif modelType == 'cnn_pose_color':
        network = build_cnn_pose_color(input_var)
    elif modelType == 'cnn_pose_norm':
        network = build_cnn_pose_norm(input_var)
    elif modelType == 'cnn_pose_large_norm':
        network = build_cnn_pose_large_norm(input_var)
    elif modelType == 'cnn_light':
        network = build_cnn_light(input_var)
    elif modelType == 'cnn_shape':
        network = build_cnn_shape(input_var)
    elif modelType == 'cnn_shape_large':
        network = build_cnn_shape_large(input_var)
    elif modelType == 'cnn_shape_medium':
        network = build_cnn_shape_medium(input_var)
    elif modelType == 'cnn_appLight':
        network = build_cnn_appLight(input_var)
    elif modelType == 'cnn_app':
        network = build_cnn_app(input_var)
    elif modelType == 'cnn_appmask':
        network = build_cnn_appmask(input_var)
    elif modelType == 'cnn_mask':
        network = build_cnn_mask(input_var)
    elif modelType == 'cnn_mask_large':
        network = build_cnn_mask_large(input_var)

    else:
        print("Unrecognized model type %r." % modelType)

    if param_values:
        lasagne.layers.set_all_param_values(network, param_values)

    return network


def get_prediction_fun(network):
    # Load the dataset

    # Prepare Theano variables for inputs and targets
    input_var = lasagne.layers.get_all_layers(network)[0].input_var

    prediction = lasagne.layers.get_output(network, deterministic=True)

    prediction_fn = theano.function([input_var], prediction)

    return prediction_fn


def get_prediction_fun_nondeterministic(network):
    # Load the dataset

    # Prepare Theano variables for inputs and targets
    input_var = lasagne.layers.get_all_layers(network)[0].input_var

    prediction = lasagne.layers.get_output(network, deterministic=False)

    prediction_fn = theano.function([input_var], prediction)

    return prediction_fn

def train_nn_h5(X_h5, trainSetVal, y_train, y_val, meanImage, network, modelType = 'cnn', num_epochs=150, saveModelAtEpoch=True, modelPath='tmp/nnmodel.pickle', param_values=[]):
    # Load the dataset

    print("Loading validation set")

    if meanImage.ndim == 2:
        meanImage = meanImage[:,:,None]

    X_val = X_h5[trainSetVal::].astype(np.float32) - meanImage.reshape([1,meanImage.shape[2], meanImage.shape[0],meanImage.shape[1]]).astype(np.float32)
    print("Ended loading validation set")

    model = {}

    model['mean'] = meanImage
    model['type'] = modelType

    if param_values:
        lasagne.layers.set_all_param_values(network, param_values)

    input_var = lasagne.layers.get_all_layers(network)[0].input_var
    target_var = T.fmatrix('targets')

    prediction = lasagne.layers.get_output(network)

    params = lasagne.layers.get_all_params(network, trainable=True)


    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    if modelType == 'cnn_mask':
        loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
        loss = loss.mean()
        test_loss = lasagne.objectives.binary_crossentropy(test_prediction, target_var)
        test_loss = test_loss.mean()
    else:
        loss = lasagne.objectives.squared_error(prediction, target_var)
        loss = loss.mean()
        test_loss = lasagne.objectives.squared_error(test_prediction,
                                                                target_var)
        test_loss = test_loss.mean()

    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accura# cy:
    val_fn = theano.function([input_var, target_var], test_loss)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:

    patience = 20
    best_valid = np.inf
    best_valid_epoch = 0
    best_weights = None
    batchSize = 128
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        slicesize = 20000
        sliceidx = 0
        for start_idx in range(0, trainSetVal, slicesize):
            sliceidx += 1
            print("Working on slice " + str(sliceidx) + " of " +  str(int(trainSetVal/slicesize)))
            X_train = X_h5[start_idx:min(start_idx + slicesize,trainSetVal)].astype(np.float32) - meanImage.reshape([1,meanImage.shape[2], meanImage.shape[0],meanImage.shape[1]]).astype(np.float32)

            for batch in iterate_minibatches(X_train, y_train[start_idx:min(start_idx + slicesize,trainSetVal)], batchSize, shuffle=True):
                # print("Batch " + str(train_batches))
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batchSize, shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1

        if val_err < best_valid:
            best_weights = lasagne.layers.get_all_param_values(network)
            if saveModelAtEpoch:
                model['params'] = best_weights
                with open(modelPath, 'wb') as pfile:
                    pickle.dump(model, pfile)
            best_valid = val_err
            best_valid_epoch = epoch

        elif best_valid_epoch + patience < epoch:
            print("Early stopping.")
            # print("Best valid loss was {:.6f} at epoch {}.".format(
            #     best_valid, best_valid_epoch))
            break

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))


    model['params'] = best_weights

    return model

def train_triplets_h5(X_h5, trainSetVal, y_train, y_val, meanImage, network, modelType = 'cnn', num_epochs=150, saveModelAtEpoch=True, modelPath='tmp/nnmodel.pickle', param_values=[]):
    # Load the dataset

    print("Loading validation set")

    if meanImage.ndim == 2:
        meanImage = meanImage[:,:,None]

    X_val = X_h5[trainSetVal::].astype(np.float32) - meanImage.reshape([1,meanImage.shape[2], meanImage.shape[0],meanImage.shape[1]]).astype(np.float32)
    print("Ended loading validation set")

    model = {}

    model['mean'] = meanImage
    model['type'] = modelType

    if param_values:
        lasagne.layers.set_all_param_values(network, param_values)

    input_var = lasagne.layers.get_all_layers(network)[0].input_var
    target_var = T.fmatrix('targets')

    prediction = lasagne.layers.get_output(network)

    params = lasagne.layers.get_all_params(network, trainable=True)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    if modelType == 'cnn_mask':
        loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
        loss = loss.mean()
        test_loss = lasagne.objectives.binary_crossentropy(test_prediction, target_var)
        test_loss = test_loss.mean()
    else:
        loss = lasagne.objectives.squared_error(prediction, target_var)
        loss = loss.mean()
        test_loss = lasagne.objectives.squared_error(test_prediction,
                                                                target_var)
        test_loss = test_loss.mean()

    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accura# cy:
    val_fn = theano.function([input_var, target_var], test_loss)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:

    patience = 20
    best_valid = np.inf
    best_valid_epoch = 0
    best_weights = None
    batchSize = 128
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        slicesize = 20000
        sliceidx = 0
        for start_idx in range(0, trainSetVal, slicesize):
            sliceidx += 1
            print("Working on slice " + str(sliceidx) + " of " +  str(int(trainSetVal/slicesize)))
            X_train = X_h5[start_idx:min(start_idx + slicesize,trainSetVal)].astype(np.float32) - meanImage.reshape([1,meanImage.shape[2], meanImage.shape[0],meanImage.shape[1]]).astype(np.float32)

            for batch in iterate_minibatches(X_train, y_train[start_idx:min(start_idx + slicesize,trainSetVal)], batchSize, shuffle=True):
                # print("Batch " + str(train_batches))
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batchSize, shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1

        if val_err < best_valid:
            best_weights = lasagne.layers.get_all_param_values(network)
            if saveModelAtEpoch:
                model['params'] = best_weights
                with open(modelPath, 'wb') as pfile:
                    pickle.dump(model, pfile)
            best_valid = val_err
            best_valid_epoch = epoch

        elif best_valid_epoch + patience < epoch:
            print("Early stopping.")
            # print("Best valid loss was {:.6f} at epoch {}.".format(
            #     best_valid, best_valid_epoch))
            break

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))


    model['params'] = best_weights

    return model

