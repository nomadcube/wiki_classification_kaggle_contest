from scipy.sparse import csc_matrix
import numpy as np

element = np.array(range(16))
row_index = np.array(range(4)).repeat(4)
col_index = np.array(range(4) * 4)
mat = csc_matrix((element, (row_index, col_index)), shape=(4, 4))

vec = csc_matrix((range(4), (range(4), [0] * 4)), shape=(4, 1))

res = mat.getcol(0) * vec.getrow(0)
for k in range(1, 4):
    res = res + mat.getcol(k) * vec.getrow(k)
print(res)
