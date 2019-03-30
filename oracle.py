'''
This is an implementation of the ORACLE-Fixed learning algorithm
'''

from matplotlib import pyplot as plt
import numpy as np

from load import load_data

import utils


def oracle():
    '''run the oracle fixed algorithm'''
    data = load_data()
    data = utils.reshape_examples(data)
    lambd = 1
    alpha = 0.3

    thetas = [
        utils.generate_random_theta(784 + 1, 32 + 1, 32, 785),
        utils.generate_random_theta(32 + 1, 10, 10, 32 + 1)
    ]

    x = list()
    plt.ion()
    for i, _ in enumerate(data):
        fig, ax = plt.subplots()
        for j in range(25):
            J, gradients = utils.non_oracle_cost(lambd, thetas, i, data[i])
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
    oracle()
