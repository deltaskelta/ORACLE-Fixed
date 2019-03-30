"""
utility functions for the ORACLE-Fixed Neural Network
"""

import math
import random
import numpy as np


def sigmoid(x):
    """return the sigmoid value of x"""
    if x < 0:
        return 1 - 1 / (1 + math.exp(x))

    return 1 / (1 + math.exp(-x))


sigmoid = np.vectorize(sigmoid)  # vectorize the sigmoid for future use


def predict(inputs, thetas):
    """make a prediction"""

    # add bias column to inputs
    inputs = add_bias_col(inputs)

    # feed forward to layer 2 and apply sigmoid
    z_values = [np.dot(inputs, np.transpose(
        thetas[0]))]  # values before the application of the sigmoid
    a_values = [sigmoid(
        z_values[0])]  # values after the application of the sigmoid

    # add bias column to a_two
    a_values[0] = add_bias_col(a_values[0])

    # feed forward to layer three
    z_values.append(np.dot(a_values[0], np.transpose(thetas[1])))
    a_values.append(sigmoid(z_values[1]))

    return a_values, z_values


def sigmoid_gradient(Z):
    """calculate the sigmoid gradient of a matrix sgm(z) .* 1 - sgm(z)"""
    return np.multiply(sigmoid(Z), np.subtract(1, sigmoid(Z)))


def generate_random_theta(l_size, l_next_size, rows, cols):
    """generate random theta values based on layer and next layer size"""
    epsilon = math.sqrt(6) / math.sqrt(l_size + l_next_size) * rand_sign()
    return np.random.uniform(low=-epsilon, high=epsilon, size=(rows, cols))


def rand_sign():
    """return a -1 or 1 randomly"""
    return -1 if random.random() < 0.5 else 1


def add_bias_col(arr):
    """add a bias column of 1's to the array at the 0 index"""
    return np.insert(arr, [0], [1], axis=1)


def non_oracle_cost(lambd, thetas, y, X):
    """
    the standard non ORACLE cost function.
    since we are training by each label in sets, the y value is just an integer label and not a one
    vs all vector
    """

    A, Z = predict(X, thetas)
    m = X.shape[0]

    # expand the scalar value into a one vs all output vector
    labels = np.zeros((1, 10))
    labels[0][y] = 1
    labels = np.repeat(labels, m, axis=0)

    # compute the gradients to return as well
    d_three = np.subtract(A[1], labels)

    d_two = np.dot(np.transpose(thetas[1])[1:, :], np.transpose(d_three))
    d_two = np.multiply(d_two, np.transpose(sigmoid_gradient(Z[0])))

    gradients = []
    gradients.append(np.dot(d_two, X) / m)  # theta 1 gradient
    gradients.append(np.dot(np.transpose(d_three), A[0])[:, 1:] / m)

    # the first term of the cost function -y * log(h(x))
    J = np.multiply(labels, np.log(A[1]))
    # the second term of the cost function 1 - y * log(h(x))
    J += np.multiply(1 - labels, np.log(1 - A[1]))
    # subtract k - l, then sum both dimensions of the matrix
    J = -np.sum(J) / m
    # add lambda regularization
    r = 0
    for i, _ in enumerate(thetas):
        r += np.sum(np.square(thetas[i][:, 1:]))
    J += (r * lambd / (2 * m))

    for i, v in enumerate(gradients):
        gradients[i] = np.add(v, lambd / m * v)

    return J, gradients


def reshape_examples(d):
    """reshape dict of sorted examples into a dict of MxN matrixes"""
    # unroll data into 1 X N vectors

    print("\nreshaping examples...")
    n = d[0][0].shape[0] * d[0][0].shape[1]

    for i in d.keys():
        new = np.array(np.ones((len(d[i]), n)))
        for j, v in enumerate(d[i]):
            # unroll the example into 1xN vector
            ex = np.reshape(v, (1, n))
            new[j] = np.multiply(new[0], ex)

        # make the dict key an MxN matrix of examples
        d[i] = new
        print("the {} examples shape: {}".format(i, d[i].shape))

    return d
