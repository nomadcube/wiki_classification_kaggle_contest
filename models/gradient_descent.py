# coding=utf-8
import math
import numpy as np


def vector_mod(x):
    """
    :param x: np.ndarray
    :return: float
    """
    return math.sqrt(np.power(x, 2).sum())


def gradient_descent(x, max_iter_time, epsilon, g):
    """
    用梯度下降法求f(x)的极小值

    :param x: np.ndarray, 极小值解的初始值
    :param max_iter_time: int, 最大迭代次数
    :param epsilon: 迭代停止条件的组成部分，当||x||小于epsilon时迭代停止
    :return: 函数对象，f(x)的梯度函数
    """
    iter_time = 1
    while iter_time < max_iter_time and vector_mod(x) >= epsilon:
        difference = g(x)
        x += (-1.0) * difference
        iter_time += 1
    return x
