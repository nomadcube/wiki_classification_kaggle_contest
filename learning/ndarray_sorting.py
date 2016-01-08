import numpy as np


def selection_sort(a, top_num):
    res = list()
    for i in range(top_num):
        res.append(np.argmax(a[i:]))
    return res


def top(arr, k):
    res = list()
    if not isinstance(arr, np.ndarray):
        raise TypeError()
    if len(arr.shape) != 2:
        raise ValueError()
    n_row, n_col = arr.shape
    for row_id in range(n_row):
        row = arr[row_id]
        res.extend(selection_sort(row, k))
    return res


if __name__ == '__main__':
    test_arr = np.array(range(15)).reshape(5, 3)
    print(top(test_arr, 2))
