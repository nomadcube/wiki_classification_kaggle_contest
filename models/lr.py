from scipy.optimize import minimize
from random import randint
import numpy as np


def norm(w, q):
    return abs(pow(sum(np.power(w, q)), 1. / q))


def sigmoid(x):
    return 1. / (1. + np.exp((-1.) * x))


def decision_function(w, x):
    return sum(w * x)


def log_likelihood(w, y, x, regularization_coef, q):
    if not isinstance(regularization_coef, float):
        raise TypeError()
    res = 0.
    for i, each_y in enumerate(y):
        if each_y == 1:
            res += np.log(sigmoid(decision_function(w, x[i])))
        else:
            res += np.log(1 - sigmoid(decision_function(w, x[i])))
    return (-1.) * res + regularization_coef * norm(w, q)


def fit(y, x, regularization_coef, q):
    init_w = np.repeat([0.1], x[0].shape[0])
    return minimize(lambda w: log_likelihood(w, y, x, regularization_coef, q), init_w).x


def predict(x, estimated_w):
    estimated_y = list()
    for i, each_x in enumerate(x):
        if sigmoid(decision_function(estimated_w, each_x)) < 0.5:
            estimated_y.append(0)
        else:
            estimated_y.append(1)
    return estimated_y


if __name__ == '__main__':
    x = [np.array([randint(1, 3) for _ in range(5)]) for _ in range(10)]
    print x
    y = [randint(0, 1) for _ in range(10)]
    print y
    estimated_w = fit(y, x, 0.3, 2)
    print estimated_w
    print fit(y, x, 0., 2)
    print predict(x, estimated_w)
    print norm([1, -2], 1)
    print norm([1, -2], 2)
