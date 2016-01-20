from array import array
from time import time

import numpy as np

TEST_SIZE = 100611105

list_start_time = time()
lst = list()
for i in range(TEST_SIZE):
    lst.append(i)
print("list append takes time: {0}".format(time() - list_start_time))

arr_start_time = time()
arr = array('i')
for i in range(TEST_SIZE):
    arr.append(i)
print("array append takes time: {0}".format(time() - arr_start_time))

np_arr_start_time = time()
np_arr = np.zeros(TEST_SIZE)
for i in range(TEST_SIZE):
    np_arr[i] = i
print("np_arr takes time: {0}".format(time() - np_arr_start_time))
