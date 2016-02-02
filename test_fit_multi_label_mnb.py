import fit_multi_label_mnb
import math


def test_fit():
    import numpy as np
    from scipy.sparse import csr_matrix
    test_y = np.array([[0], [0], [1], [1], [0],
                       [0], [0], [1], [1], [1],
                       [1], [1], [1], [1], [0]])
    element = [1.] * 30
    row_index = list()
    for i in range(15):
        row_index.extend([i] * 2)
    col_index = [0, 3,
                 0, 4,
                 0, 4,
                 0, 3,
                 0, 3,
                 1, 3,
                 1, 4,
                 1, 4,
                 1, 5,
                 1, 5,
                 2, 5,
                 2, 4,
                 2, 4,
                 2, 5,
                 2, 5]
    test_x = csr_matrix((element, (row_index, col_index)), shape=(15, 6))
    m = fit_multi_label_mnb.fit(test_y, test_x)
    assert m.nnz == 14
    assert m.shape == (2, 7)
    assert abs(m[0, 0] - math.log(0.5)) < 1e-6
    assert abs(m[0, 1] - math.log(0.333333333333)) < 1e-6
    assert abs(m[0, 2] - math.log(0.166666666667)) < 1e-6
    assert abs(m[0, 3] - math.log(0.5)) < 1e-6
    assert abs(m[0, 4] - math.log(0.333333333333)) < 1e-6
    assert abs(m[0, 5] - math.log(0.166666666667)) < 1e-6
    assert abs(m[1, 0] - math.log(0.222222222222)) < 1e-6
    assert abs(m[1, 1] - math.log(0.333333333333)) < 1e-6
    assert abs(m[1, 2] - math.log(0.444444444444)) < 1e-6
    assert abs(m[1, 3] - math.log(0.111111111111)) < 1e-6
    assert abs(m[1, 4] - math.log(0.444444444444)) < 1e-6
    assert abs(m[1, 5] - math.log(0.444444444444)) < 1e-6
