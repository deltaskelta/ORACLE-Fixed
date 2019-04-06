"""
an implementation of the ORACLE-Fixed learning algorithm
"""

import scipy.optimize as op
#from matplotlib import pyplot as plt
import numpy as np
from load import load_data
import utils


def validation_predict(val_set, thetas, i):
    """cross validation set prediction"""
    # predict everything until i to see how it does on those examples
    print("")
    for j in range(i + 1):
        A, _ = utils.predict(utils.add_bias_col(val_set[j]), thetas)
        predicted = np.argmax(A[1], axis=1)
        accuracy = np.mean(np.int8(predicted == j)) * 100
        print("for task {}, accuracy: {} for digit {}".format(i, accuracy, j))


# pylint: disable=too-many-locals
def plain_nn(data, order, theta_vec, val):
    """
    train a neural network that learns the entire dataset at once as a base to judge continous
    learning algorithms against
    """
    lambd = 1

    examples = np.array(data[0])
    labels = np.array(utils.make_labels(0, data[0].shape[0]))
    for i, _ in enumerate(data, start=0):
        if i != 0:
            y = order[i]
            m = data[y].shape[0]
            labels = np.append(labels, utils.make_labels(y, 200), axis=0)
            examples = np.append(examples, data[y][:200, :], axis=0)

    res = op.minimize(
        utils.cost,
        theta_vec,
        method='CG',
        jac=True,
        options=dict({
            "disp": True,
            "maxiter": 100,
        }),
        args=(m, lambd, labels, examples))

    validation_predict(val, utils.unravel_theta(res["x"]), 9)


# pylint: disable=too-many-locals
def plain_forgetful_nn(data, order, theta_vec, val):
    """
    train a network to learn new tasks with plain l2 regularization, forgets everything 
    previous when learning a new task
    """
    lambd = 1

    for i, _ in enumerate(data, start=0):
        y = order[i]
        m = data[y].shape[0]

        res = op.minimize(
            utils.cost,
            theta_vec,
            method='CG',
            jac=True,
            options=dict({
                "disp": True,
            }),
            args=(m, lambd, y, data[y]))

        theta_vec = res["x"].copy()
        validation_predict(val, utils.unravel_theta(theta_vec), i)


# pylint: disable=too-many-locals
def theta_distance_reg(data, order, theta_vec, val):
    """
    train a NN with l2 regularization on thetas^l - thetas^l-1, which should place a higher cost
    when theta changes a more from the previous theta
    """
    lambd = 100
    for i, _ in enumerate(data, start=0):
        y = order[i]
        m = data[y].shape[0]
        labels = utils.make_labels(i, m)

        # If we are on the first loop we want regular l2 normalization because we want to
        # enforce that Thetas should be as low valued as possible. If it is not in loop one
        # then we can just do theta difference l2 normalization which is the goal
        if i == 0:
            res = op.minimize(
                utils.cost,
                theta_vec,
                method='CG',
                jac=True,
                options=dict({
                    "disp": True,
                    "maxiter": 50,
                }),
                args=(m, 1, labels, data[y]))
        else:
            res = op.minimize(
                utils.theta_diff_cost,
                theta_vec,
                method='CG',
                jac=True,
                options=dict({
                    "disp": True,
                    "maxiter": 50,
                }),
                args=(theta_vec.copy(), m, lambd, labels, data[y]))

        # set current theta vec for next loop complexity calc
        theta_vec = res["x"].copy()
        thetas = utils.unravel_theta(res["x"])

        validation_predict(val, thetas, i)


# pylint: disable=too-many-locals
def oracle_fixed(data, order, theta_vec, val):
    """
    train a neural network according to the ORACLE-Fixed algorithm
    """
    lambd = 0.1
    lambd2 = 0.1
    alpha = 0.1

    for i, _ in enumerate(data, start=0):
        y = order[i]
        m = data[y].shape[0]

        # If we are on the first loop we want regular l2 normalization because we want to
        # enforce that Thetas should be as low valued as possible. If it is not in loop one
        # then we can just do theta difference l2 normalization which is the goal
        if i == 0:
            res = op.minimize(
                utils.cost,
                theta_vec,
                method='CG',
                jac=True,
                options=dict({
                    "disp": True,
                }),
                args=(m, 1, y, data[y]))
        else:
            shared_prev = theta_vec.copy()
            unrolled = utils.unravel_theta(shared_prev)

            ps = list()
            qs = list()
            for j, th in enumerate(unrolled):
                tau, p, q = utils.make_tau(th)
                unrolled[j] += tau
                ps.append(p)
                qs.append(q)

            rolled = utils.ravel_theta(unrolled)

            res = op.minimize(
                utils.oracle_fixed_cost,
                rolled,
                method='CG',
                jac=True,
                options=dict({
                    "disp": True,
                }),
                args=(shared_prev, ps, qs, m, lambd, lambd2, y, data[y]))

        theta_vec = res["x"].copy()

        # set current theta vec for next loop complexity calc
        thetas = utils.unravel_theta(theta_vec.copy())
        validation_predict(val, thetas, i)


if __name__ == '__main__':
    d, v = load_data()
    d = utils.reshape_examples(d)
    v = utils.reshape_examples(v)

    tv = utils.ravel_theta([
        utils.generate_random_theta(utils.T1[1] + 1, utils.T1[0] + 1,
                                    utils.T1[0], utils.T1[1]),
        utils.generate_random_theta(utils.T2[1] + 1, utils.T2[0] + 1,
                                    utils.T2[0], utils.T2[1])
    ])

    theta_distance_reg(d, list(d), tv, v)
    plain_nn(d, list(d), tv, v)
    plain_forgetful_nn(d, list(d), tv, v)
    #oracle_fixed(d, list(d), tv, v)
