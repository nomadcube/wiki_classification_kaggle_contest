import fit_multi_label_mnb
import predict_multi_label_mnb


def test_fit():
    import numpy as np
    from scipy.sparse import csr_matrix
    import math
    test_y = np.array([[314523, 165538, 416827], [21631], [76255, 335416]])
    test_x = csr_matrix(([1.0, 4.0, 1.0, 1.0, 1.0, 1.0],
                         ([0, 1, 1, 1, 2, 2], [1250536, 1095476, 805104, 634175, 1250536, 805104])))
    mn_param = fit_multi_label_mnb.fit(test_y, test_x)
    assert mn_param.shape == (416828, 1250537 + 1)
    assert mn_param.nnz == 13
    assert mn_param[21631, 634175] == math.log(1. / 6.)
    assert mn_param[21631, 1095476] == math.log(4. / 6.)
    assert mn_param[21631, 805104] == math.log(1. / 6.)
    assert mn_param[76255, 1250536] == math.log(1. / 2.)
    assert mn_param[76255, 805104] == math.log(1. / 2.)
    assert mn_param[335416, 1250536] == math.log(1. / 2.)
    assert mn_param[335416, 805104] == math.log(1. / 2.)
    assert mn_param[314523, 1250537] == math.log(1. / 6.)
    assert mn_param[165538, 1250537] == math.log(1. / 6.)
    assert mn_param[416827, 1250537] == math.log(1. / 6.)
    assert mn_param[21631, 1250537] == math.log(1. / 6.)
    assert mn_param[76255, 1250537] == math.log(1. / 6.)
    assert mn_param[335416, 1250537] == math.log(1. / 6.)


def test_log_likelihood():
    import numpy as np
    from scipy.sparse import csr_matrix
    import math
    test_y = np.array([[314523, 165538, 416827], [21631], [76255, 335416]])
    test_x = csr_matrix(([1.0, 4.0, 1.0, 1.0, 1.0, 1.0],
                         ([0, 1, 1, 1, 2, 2], [1250536, 1095476, 805104, 634175, 1250536, 805104])))
    model = fit_multi_label_mnb.fit(test_y, test_x)
    test_x = fit_multi_label_mnb.add_unit_column(test_x.tocsc())
    ll = predict_multi_label_mnb._log_likelihood(test_x, model)
    assert ll.shape == (416828, 3)
    assert ll.nnz == 18
    assert ll[314523, 0] == math.log(1. / 6.)
    assert ll[314523, 1] == math.log(1. / 6.)
    assert ll[314523, 2] == math.log(1. / 6.)
    assert ll[165538, 0] == math.log(1. / 6.)
    assert ll[165538, 1] == math.log(1. / 6.)
    assert ll[165538, 2] == math.log(1. / 6.)
    assert ll[416827, 0] == math.log(1. / 6.)
    assert ll[416827, 1] == math.log(1. / 6.)
    assert ll[416827, 2] == math.log(1. / 6.)
    assert ll[21631, 0] == math.log(1. / 6.)
    assert ll[21631, 1] == math.log(1. / 6.) + 1. * math.log(1. / 6.) + 4. * math.log(4. / 6.) + 1 * math.log(1. / 6.)
    assert ll[21631, 2] == math.log(1. / 6.) + 1. * math.log(1. / 6.)
    assert ll[76255, 0] == math.log(1. / 6.) + 1. * math.log(1. / 2.)
    assert ll[76255, 1] == math.log(1. / 6.) + 1. * math.log(1. / 2.)
    assert abs(ll[76255, 2] - (math.log(1. / 6.) + 1. * math.log(1. / 2.) + 1. * math.log(1. / 2.))) < 0.00001
    assert ll[335416, 0] == math.log(1. / 6.) + 1. * math.log(1. / 2.)
    assert ll[335416, 1] == math.log(1. / 6.) + 1. * math.log(1. / 2.)
    assert abs(ll[335416, 2] - (math.log(1. / 6.) + 1. * math.log(1. / 2.) + 1. * math.log(1. / 2.))) < 0.00001


def test_block_x():
    from scipy.sparse import csr_matrix
    test_x = csr_matrix(([1.0, 4.0, 1.0, 1.0, 1.0, 1.0],
                         ([0, 1, 1, 1, 2, 2], [1250536, 1095476, 805104, 634175, 1250536, 805104])))
    all_block_x = [i for i in predict_multi_label_mnb._block_x(test_x, 1)]
    assert len(all_block_x) == 3
    assert all_block_x[0].shape == (1, 1250537)
    assert all_block_x[1].shape == (1, 1250537)
    assert all_block_x[2].shape == (1, 1250537)
    assert all_block_x[0].nnz == 1
    assert all_block_x[1].nnz == 3
    assert all_block_x[2].nnz == 2


def test_predict():
    import numpy as np
    from scipy.sparse import csr_matrix
    test_y = np.array([[314523, 165538, 416827], [21631], [76255, 335416]])
    test_x = csr_matrix(([1.0, 4.0, 1.0, 1.0, 1.0, 1.0],
                         ([0, 1, 1, 1, 2, 2], [1250536, 1095476, 805104, 634175, 1250536, 805104])))
    model = fit_multi_label_mnb.fit(test_y, test_x)
    predict_res = predict_multi_label_mnb.predict(test_x, model)
    assert len(predict_res) == 3
    assert predict_res[0] == [165538]
    assert predict_res[1] == [165538]
    assert predict_res[2] == [165538]
