from scipy.optimize import minimize
from random import randint
import numpy as np


def p_norm(w, p):
    return abs(pow(sum(np.power(w, p)), 1. / p))


def sigmoid(x):
    return 1. / (1. + np.exp((-1.) * x))


def discrimination(w, x):
    return sum(w * x)


def empirical_risk(w, y, x):
    res = 0.
    for i, each_y in enumerate(y):
        if each_y == 1:
            res += np.log(sigmoid(discrimination(w, x[i])))
        else:
            res += np.log(1 - sigmoid(discrimination(w, x[i])))
    return (-1.) * res


class LR:
    def __init__(self, regularization_coefficient, p):
        self._regularization_coefficient = regularization_coefficient
        self._p = p
        self.w = None

    def fit(self, y, x):
        self.w = self._solve(y, x, init_w=[0.1, 0.1, 0.1, 0.1, 0.1])

    def predict(self, x):
        estimated_y = list()
        for i, each_x in enumerate(x):
            if sigmoid(discrimination(self.w, each_x)) < 0.5:
                estimated_y.append(0)
            else:
                estimated_y.append(1)
        return estimated_y

    def _solve(self, y, x, init_w):
        return minimize(
            lambda w: empirical_risk(w, y, x) + float(self._regularization_coefficient) * p_norm(w, self._p),
            init_w).x


if __name__ == '__main__':
    x = [np.array([randint(1, 3) for _ in range(5)]) for _ in range(10)]
    print x
    y = [randint(0, 1) for _ in range(10)]
    print y
    lr = LR(0, 2)
    lr.fit(y, x)
    print lr.w
    print lr.predict(x)

    print p_norm([1, -2], 1)
    print p_norm([1, -2], 2)
