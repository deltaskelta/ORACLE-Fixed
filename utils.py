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
    z_values = [np.dot(inputs, thetas[0].T)]
    a_values = [sigmoid(z_values[0])]

    # add bias column to a_two
    a_values[0] = add_bias_col(a_values[0])

    # feed forward to layer three
    z_values.append(np.dot(a_values[0], thetas[1].T))
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


# pylint: disable=too-many-arguments
def cost(m, lambd, thetas, y, X, complx):
    """
    the standard non ORACLE cost function.
    since we are training by each label in sets, the y value is just an integer label and not a one
    vs all vector
    """

    A, Z = predict(X, thetas)

    # expand the scalar value into a one vs all output vector
    labels = np.zeros((1, 10))
    labels[0][y] = 1
    labels = np.repeat(labels, m, axis=0)

    # compute the gradients to return as well
    d_three = np.subtract(A[1], labels)

    d_two = np.dot(thetas[1].T[1:, :], d_three.T)
    d_two = np.multiply(d_two, sigmoid_gradient(Z[0]).T)

    gradients = []
    gradients.append(np.dot(d_two, X) / m)  # theta 1 gradient
    gradients.append(np.dot(d_three.T, A[0])[:, 1:] / m)

    # the first term of the cost function -y * log(h(x))
    J = np.multiply(labels, np.log(A[1]))
    # the second term of the cost function 1 - y * log(h(x))
    J += np.multiply(1 - labels, np.log(1 - A[1]))
    # subtract k - l, then sum both dimensions of the matrix
    J = -np.sum(J) / m
    # add lambda regularization (complexity term)
    J += complx

    for i, v in enumerate(gradients):
        gradients[i] = np.add(v, lambd / m * v)

    return J, gradients


def l2_normalization(thetas, lambd, m):
    """l2 normalization of theta values"""

    r = 0
    for i, _ in enumerate(thetas):
        r += np.sum(np.square(thetas[i][:, 1:]))
    return r * lambd / (2 * m)


def theta_difference_l2_normalization(thetas, thetas_prev, lambd):
    """l2 normalization based on the difference in theta values"""
    r = 0
    for i, _ in enumerate(thetas):
        t = thetas[i] - thetas_prev[i]
        r += np.sum(np.square(t[:, 1:]))
    return r * lambd


def make_tau(sigma):
    """make the tau parameter for the ORACLE-Fixed algorithm"""

    print("theta size: ", sigma.shape)
    u, s, v = np.linalg.svd(sigma, full_matrices=True)
    print("u size: ", u.shape)
    print("sigma size: ", s.shape)
    print("v.shape: ", v.shape)

    p = np.dot(u, np.sqrt(np.diag(s)))
    print("p size: ", p.shape)

    # TODO: the zeros term needs to expand theta to be compatible with p
    # you can check p's size and add whatever is needed

    # it should be the same size as theta (p * q.T) and p and q should be sparse
    # it should a
    s = np.diag(s)
    S = np.append(s, np.zeros((v.shape[0] - u.shape[0], s.shape[0])), axis=0)
    #S = np.append(s, np.zeros(()))
    print("S shape: ", S.shape)
    print(p)
    q = np.dot(v, np.sqrt(S))
    print(q)

    return np.dot(p, q.T)


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
