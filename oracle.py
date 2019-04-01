"""
This is an implementation of the ORACLE-Fixed learning algorithm
"""

from matplotlib import pyplot as plt
import numpy as np

from load import load_data

import utils


# pylint: disable=too-many-locals
def plain_nn(data, order):
    """
    train a neural network with l2 theta regularization that forgets everything when
    learning a new task
    """
    lambd = 1
    alpha = 0.3

    thetas = [
        utils.generate_random_theta(784 + 1, 32 + 1, 32, 785),
        utils.generate_random_theta(32 + 1, 10, 10, 32 + 1)
    ]

    x = list()
    plt.ion()
    for i, _ in enumerate(data, start=0):
        _, ax = plt.subplots()
        for j in range(25):
            y = order[i]
            m = data[y].shape[0]

            complx = utils.l2_normalization(thetas, lambd, m)
            J, gradients = utils.cost(m, lambd, thetas, y, data[i], complx)

            x.append(J)
            ax.plot(x)
            plt.pause(0.00001)
            plt.draw()

            for k, _ in enumerate(thetas):
                thetas[k] = thetas[k][:, 1:]
                thetas[k] -= gradients[k] * alpha
                thetas[k] = utils.add_bias_col(thetas[k])

        plt.clf()
        x = list()

        # predict everything until i to see how it does on those examples
        for j in range(i + 1):
            A, _ = utils.predict(data[j], thetas)
            predicted = np.argmax(A[1], axis=1)
            accuracy = np.mean(np.int8(predicted == j)) * 100
            print("for task {}, standard NN accuracy: {} for digit {}".format(
                i, accuracy, j))


def theta_distance_reg(data, order):
    """
    train a NN with l2 regularization on thetas^l - thetas^l-1, which should place a higher cost
    when theta changes a lost from the previous theta
    """

    lambd = 0.3
    alpha = 0.1

    thetas = [
        utils.generate_random_theta(784 + 1, 32 + 1, 32, 785),
        utils.generate_random_theta(32 + 1, 10, 10, 32 + 1)
    ]

    x = list()
    plt.ion()
    for i, _ in enumerate(data, start=0):
        _, ax = plt.subplots()

        thetas_prev = [t.copy() for t in thetas]
        for j in range(15):
            y = order[i]
            m = data[y].shape[0]

            # If we are on the first loop we want regular l2 normalization because we want to
            # enforce that Thetas should be as low valued as possible. If it is not in loop one
            # then we can just do theta difference l2 normalization which is the goal
            if i == 0:
                complx = utils.l2_normalization(
                    thetas, 1, m)  # has a different lambda value
            else:
                complx = utils.theta_difference_l2_normalization(
                    thetas, thetas_prev, lambd)
                print("complx is: {}".format(complx))

            J, gradients = utils.cost(m, lambd, thetas, y, data[y], complx)

            x.append(J)
            ax.plot(x)
            plt.pause(0.00001)
            plt.draw()

            for k, _ in enumerate(thetas):
                thetas[k] = thetas[k][:, 1:]
                thetas[k] -= gradients[k] * alpha
                thetas[k] = utils.add_bias_col(thetas[k])

        plt.clf()
        x = list()

        # predict everything until i to see how it does on those examples
        for j in range(i + 1):
            A, _ = utils.predict(data[j], thetas)
            predicted = np.argmax(A[1], axis=1)
            accuracy = np.mean(np.int8(predicted == j)) * 100
            print("for task {}, standard NN accuracy: {} for digit {}".format(
                i, accuracy, j))


if __name__ == '__main__':
    _data = load_data()
    _data = utils.reshape_examples(_data)

    theta_distance_reg(_data, list(_data))
    plain_nn(_data, list(_data))
