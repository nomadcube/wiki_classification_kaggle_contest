import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 2]])
print a.shape


def get_second(a):
    return -a


get_second_array = np.frompyfunc(get_second, 1, 1)
print np.add.reduce(a, 1)
