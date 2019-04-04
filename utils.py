"""
utility functions for the ORACLE-Fixed Neural Network
"""

import math
import numpy as np

# both will be +1 column for bias
T1 = (32, 784 + 1)
T2 = (10, 32 + 1)


def sigmoid(x):
    """return the sigmoid value of x"""
    if x < 0:
        return 1 - 1 / (1 + math.exp(x))

    return 1 / (1 + math.exp(-x))


sigmoid = np.vectorize(sigmoid)  # vectorize the sigmoid for future use


def predict(inputs, thetas):
    """make a prediction"""

    # feed forward to layer 2 and apply sigmoid
    z_values = [np.dot(inputs, thetas[0].T)]  # 1x3 * (5x3).T = 1x5
    a_values = [sigmoid(z_values[0])]  # 1x5
    # add bias column to a_two
    a_values[0] = add_bias_col(a_values[0])  # 1x6
    # feed forward to layer three
    # 1x6 * (1x6).T = 1x1
    z_values.append(np.dot(a_values[0], thetas[1].T))
    a_values.append(sigmoid(z_values[1]))  #1x1

    return a_values, z_values


def sigmoid_gradient(Z):
    """calculate the sigmoid gradient of a matrix sgm(z) .* 1 - sgm(z)"""
    sgd = sigmoid(Z)
    return sgd * (1 - sgd)


def generate_random_theta(l_size, l_next_size, rows, cols):
    """generate random theta values based on layer and next layer size"""
    epsilon = math.sqrt(6) / math.sqrt(l_size + l_next_size)
    arr = np.random.uniform(low=-epsilon, high=epsilon, size=(rows, cols))

    # add bias column
    arr[:, 0] = 1
    return arr


def add_bias_col(arr):
    """add a bias column of 1's to the array at the 0 index"""
    return np.insert(arr, [0], [1], axis=1)


def make_labels(y, m):
    """make one vs all labels from a scalar value"""

    labels = np.zeros((1, T2[0]))
    labels[0][y] = 1

    return np.repeat(labels, m, axis=0)


def make_tau(sigma):
    """make the tau parameter for the ORACLE-Fixed algorithm"""

    u, s, v = np.linalg.svd(sigma, full_matrices=True)
    p = np.dot(u, np.sqrt(np.diag(s)))

    # it should be the same size as theta (p * q.T) and p and q should be sparse
    # it should a
    s = np.diag(s)
    S = np.append(s, np.zeros((v.shape[0] - u.shape[0], s.shape[0])), axis=0)

    q = np.dot(v, np.sqrt(S))

    return np.dot(p, q.T), p, q


def log_likelihood(labels, A, m):
    """the log likelihood portion of the cost function"""
    # the first term of the cost function -y * log(h(x)), second term 1 - y * log(h(x))
    J = -labels * np.log(A[1])
    J -= (1 - labels) * np.log(1 - A[1])
    return np.sum(J) / m


def l2_normalization(thetas, lambd, m):
    """l2 normalization of theta values"""

    r = 0
    for _, v in enumerate(thetas):
        r += np.sum(np.square(v[:, 1:]))
    return r * lambd / (2 * m)


def theta_difference_l2_normalization(thetas, thetas_prev, m, lambd):
    """l2 normalization based on the difference in theta values"""

    r = 0
    for i, _ in enumerate(thetas):
        t = thetas[i] - thetas_prev[i]
        r += np.sum(np.square(t[:, 1:]))
    return r * lambd


def cost(thetas, m, lambd, y, X):
    """
    the standard non ORACLE cost function.
    since we are training by each label in sets, the y value is just an integer label and not a one
    vs all vector
    """

    X = add_bias_col(X)  # now 1x3
    thetas = unravel_theta(thetas)  # [5x3, 1x6]

    # A = [1x6, 1x1] Z = [1x5, 1x1]
    A, Z = predict(X, thetas)
    labels = make_labels(y, m)

    # compute the deltas and gradients to return as well
    d_three = A[1] - labels  # 1x1
    # (1x6).T * 1x1 = 6x1 --> 5x1 (from indexing)
    d_two = np.dot(thetas[1].T, d_three.T)[1:, :] * sigmoid_gradient(Z[0]).T

    gradients = [
        np.dot(d_two, X) / m,  # 5x1 * 1x3 = 5x3
        np.dot(d_three.T, A[0]) / m  # 1x1 * 1x6 = 1x6
    ]

    J = log_likelihood(labels, A, m)
    J += l2_normalization(thetas, lambd, m)  # lambda complexity penalty

    for i, _ in enumerate(gradients):
        gradients[i][:, 1:] += lambd / m * thetas[i][:, 1:]

    return J, ravel_theta(gradients)


# pylint: disable=too-many-locals, too-many-arguments
def theta_diff_cost(thetas, thetas_prev, m, lambd, y, X):
    """
    the standard non ORACLE cost function.
    since we are training by each label in sets, the y value is just an integer label and not a one
    vs all vector
    """

    thetas_mag = np.sqrt(np.sum(np.square(thetas)))
    thetas_pre_mag = np.sqrt(np.sum(np.square(thetas_prev)))
    print("thetas: ", thetas_mag, " thetas_prev: ", thetas_pre_mag)

    # TODO: should this difference be calculated ignoring the bias theta units? if so, this needs to move to
    # a calulation that happens after the theta values are unrolled
    err = np.sqrt(np.sum(np.square(thetas[:, 1:] - thetas_prev[:, 1:])))
    print("calculated error is: ", err)

    X = add_bias_col(X)
    thetas = unravel_theta(thetas)
    thetas_prev = unravel_theta(thetas_prev)

    A, Z = predict(X, thetas)
    labels = make_labels(y, m)

    # compute the deltas and gradients to return as well
    d_three = A[1] - labels
    d_two = np.dot(thetas[1].T, d_three.T)[1:, :] * sigmoid_gradient(Z[0]).T

    gradients = [
        np.dot(d_two, X) / m,  # theta 1 gradient
        np.dot(d_three.T, A[0]) / m
    ]

    # add lambda regularization (complexity term)
    J = log_likelihood(labels, A, m)
    diff = theta_difference_l2_normalization(thetas, thetas_prev, m, lambd)
    J += diff
    print("cost: {} penalization: {}".format(J, diff))

    for i, _ in enumerate(gradients):
        gradients[i][:, 1:] += 2 * lambd * thetas[i][:, 1:]

    print(J)

    return J, ravel_theta(gradients)


def l1_tau(lambd, p, q):
    """l1 normalization of tau factors"""
    return lambd * (np.sum(np.abs(p)) + np.sum(np.abs(q)))


# pylint: disable=too-many-locals, too-many-arguments
def oracle_fixed_cost(thetas, shared, ps, qs, m, lambd, lambd2, y, X):
    """
    shared cost for the ORACLE-Fixed shared parameter optimization
    """

    X = add_bias_col(X)
    thetas = unravel_theta(thetas)
    thetas_prev = unravel_theta(shared)

    A, Z = predict(X, thetas)  # make prediction with theta + tau
    labels = make_labels(y, m)

    # compute the gradients to return as well
    d_three = np.subtract(A[1], labels)

    d_two = np.dot(thetas[1].T, d_three.T)[1:, :]
    d_two = np.multiply(d_two, sigmoid_gradient(Z[0]).T)

    gradients = []
    gradients.append(np.dot(d_two, X) / m)  # theta 1 gradient
    gradients.append(np.dot(d_three.T, A[0]) / m)

    # add lambda regularization (complexity term)
    J = log_likelihood(labels, A, m)
    J += theta_difference_l2_normalization(thetas, thetas_prev, m, lambd)
    for i, _ in enumerate(ps):
        J += l1_tau(lambd2, ps[i], qs[i])

    for i, v in enumerate(gradients):
        gradients[i][:, 1:] = np.add(v[:, 1:], lambd / m * v[:, 1:])

    return J, ravel_theta(gradients)


def callback(xk):
    """callback made on every loop of cost by optimization func"""
    return True


def ravel_theta(thetas):
    """unroll thetas into a big vector"""
    r = np.array([])
    for t in thetas:
        r = np.append(r, t.ravel())
    return r


def unravel_theta(theta_vec):
    """rehshape the thetas into matrices"""
    t1 = T1[0] * T1[1]
    t2 = T2[0] * T2[1]

    theta1_vals = theta_vec[0:t1]
    theta2_vals = theta_vec[t1:t1 + t2]

    return [theta1_vals.reshape(T1), theta2_vals.reshape(T2)]


def check_nn_gradients(theta_vec, y, data):
    """check gradient implementation"""
    epsilon = 0.00001
    m = data.shape[0]

    gradApprox = theta_vec.copy()  # because it is the same size
    for i, _ in enumerate(theta_vec):
        thetaPlus = theta_vec.copy()
        thetaPlus[i] += epsilon
        thetaMinus = theta_vec.copy()
        thetaMinus[i] -= epsilon

        jPlus, _ = cost(thetaPlus, m, 1, y, data)
        jMinus, _ = cost(thetaMinus, m, 1, y, data)

        gradApprox[i] = (jPlus - jMinus) / (2 * epsilon)

    _, grad = cost(theta_vec, m, 1, y, data)

    l2_g = np.sqrt(np.sum(np.square(grad)))
    l2ga = np.sqrt(np.sum(np.square(gradApprox)))
    print("l2g: ", l2_g, " l2ga: ", l2ga)

    err = np.sqrt(np.sum(np.square(grad - gradApprox)))
    print("calculated error is: ", err)

    for i, _ in enumerate(grad):
        print("grad: {} approximation: {} diff: {}".format(
            grad[i], gradApprox[i], grad[i] - gradApprox[i]))


def check_td_gradients(theta_vec, y, data):
    """check gradient implementation"""
    epsilon = 0.00001
    m = data.shape[0]
    lambd = 0.001

    gradApprox = theta_vec.copy()  # because it is the same size
    for i, _ in enumerate(theta_vec):
        thetaPlus = theta_vec.copy()
        thetaPlus[i] += epsilon
        thetaMinus = theta_vec.copy()
        thetaMinus[i] -= epsilon

        jPlus, _ = theta_diff_cost(thetaPlus, thetaPlus, m, lambd, y, data)
        jMinus, _ = theta_diff_cost(thetaMinus, thetaMinus, m, lambd, y, data)

        gradApprox[i] = (jPlus - jMinus) / (2 * epsilon)

    _, grad = theta_diff_cost(theta_vec, theta_vec, m, lambd, y, data)

    l2_g = np.sqrt(np.sum(np.square(grad)))
    l2ga = np.sqrt(np.sum(np.square(gradApprox)))
    print("l2g: ", l2_g, " l2ga: ", l2ga)

    err = np.sqrt(np.sum(np.square(grad - gradApprox)))
    print("calculated error is: ", err)

    for i, _ in enumerate(grad):
        print("grad: {} approximation: {} diff: {}".format(
            grad[i], gradApprox[i], grad[i] - gradApprox[i]))


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

            if i == 0 and j == 0:
                new[j] = np.multiply(new[0], ex)
            else:
                new[j] = ex

        # make the dict key an MxN matrix of examples, regularize the data to be between 0 and 1
        d[i] = new
        print("the {} examples shape: {}".format(i, d[i].shape))

    # scale the features (x - avg(x) / stdev(x))
    together = np.array([])
    for i in d.keys():
        together = np.append(together, d[i].ravel(), axis=0)

    mean = np.mean(together)
    stddev = np.std(together)
    print("max: ", np.max(together))
    print("min: ", np.min(together))
    print("mean: ", mean, " stddev: ", stddev)

    for i in d.keys():
        d[i] = (d[i] - mean) / stddev

    return d
