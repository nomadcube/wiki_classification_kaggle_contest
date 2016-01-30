import fit_multi_label_mnb


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
    prior_param = fit_multi_label_mnb.fit(test_y, test_x)[0]
    mn_param = fit_multi_label_mnb.fit(test_y, test_x)[1]
    assert prior_param.shape == (1, 2)
    assert mn_param.shape == (2, 6)
    assert mn_param.nnz == 12
    assert mn_param[0, 0] == 0.5
    assert round(mn_param[0, 1], 12) == 0.333333333333
    assert round(mn_param[0, 2], 12) == 0.166666666667
    assert round(mn_param[0, 3], 12) == 0.5
    assert round(mn_param[0, 4], 12) == 0.333333333333
    assert round(mn_param[0, 5], 12) == 0.166666666667
    assert round(mn_param[1, 0], 12) == 0.222222222222
    assert round(mn_param[1, 1], 12) == 0.333333333333
    assert round(mn_param[1, 2], 12) == 0.444444444444
    assert round(mn_param[1, 3], 12) == 0.111111111111
    assert round(mn_param[1, 4], 12) == 0.444444444444
    assert round(mn_param[1, 5], 12) == 0.444444444444
