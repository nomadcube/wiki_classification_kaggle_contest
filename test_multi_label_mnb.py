import fit_multi_label_mnb
import predict_multi_label_mnb


def test_fit():
    import numpy as np
    from scipy.sparse import csr_matrix
    import math
    test_y = np.array([[314523, 165538, 416827], [21631], [76255, 335416]])
    test_x = csr_matrix(([1.0, 4.0, 1.0, 1.0, 1.0, 1.0],
                         ([0, 1, 1, 1, 2, 2], [1250536, 1095476, 805104, 634175, 1250536, 805104])))
    post_prob_y, mn_param = fit_multi_label_mnb.fit(test_y, test_x)
    assert post_prob_y.shape == (1, 416828)
    assert mn_param.shape == (416828, 1250537)
    assert post_prob_y.nnz == 6
    assert post_prob_y[0, 314523] == math.log(1. / 6.)
    assert post_prob_y[0, 165538] == math.log(1. / 6.)
    assert post_prob_y[0, 416827] == math.log(1. / 6.)
    assert post_prob_y[0, 21631] == math.log(1. / 6.)
    assert post_prob_y[0, 76255] == math.log(1. / 6.)
    assert post_prob_y[0, 335416] == math.log(1. / 6.)
    assert mn_param.nnz == 10
    assert mn_param[314523, 1250536] == 0
    assert mn_param[165538, 1250536] == 0
    assert mn_param[416827, 1250536] == 0
    assert mn_param[21631, 634175] == math.log(1. / 6.)
    assert mn_param[21631, 1095476] == math.log(4. / 6.)
    assert mn_param[21631, 805104] == math.log(1. / 6.)
    assert mn_param[76255, 1250536] == math.log(1. / 2.)
    assert mn_param[76255, 805104] == math.log(1. / 2.)
    assert mn_param[335416, 1250536] == math.log(1. / 2.)
    assert mn_param[335416, 805104] == math.log(1. / 2.)


def test_log_likelihood():
    import numpy as np
    from scipy.sparse import csr_matrix
    import math
    test_y = np.array([[314523, 165538, 416827], [21631], [76255, 335416]])
    test_x = csr_matrix(([1.0, 4.0, 1.0, 1.0, 1.0, 1.0],
                         ([0, 1, 1, 1, 2, 2], [1250536, 1095476, 805104, 634175, 1250536, 805104])))
    model = fit_multi_label_mnb.fit(test_y, test_x)
    ll = predict_multi_label_mnb._log_likelihood(test_x, model)
    assert ll.shape == (416828, 3)
    assert ll.nnz == 8
    assert ll[21631, 1] == math.log(1. / 6.) + 1. * math.log(1. / 6.) + 4. * math.log(4. / 6.) + 1 * math.log(1. / 6.)
    assert ll[21631, 2] == math.log(1. / 6.) + 1. * math.log(1. / 6.)
    assert ll[76255, 0] == math.log(1. / 6.) + 1. * math.log(1. / 2.)
    assert ll[76255, 1] == math.log(1. / 6.) + 1. * math.log(1. / 2.)
    assert abs(ll[76255, 2] - (math.log(1. / 6.) + 1. * math.log(1. / 2.) + 1. * math.log(1. / 2.))) < 0.00001
    assert ll[335416, 0] == math.log(1. / 6.) + 1. * math.log(1. / 2.)
    assert ll[335416, 1] == math.log(1. / 6.) + 1. * math.log(1. / 2.)
    assert abs(ll[335416, 2] - (math.log(1. / 6.) + 1. * math.log(1. / 2.) + 1. * math.log(1. / 2.))) < 0.00001


def test_predict():
    import numpy as np
    from scipy.sparse import csr_matrix
    test_y = np.array([[314523, 165538, 416827], [21631], [76255, 335416]])
    test_x = csr_matrix(([1.0, 4.0, 1.0, 1.0, 1.0, 1.0],
                         ([0, 1, 1, 1, 2, 2], [1250536, 1095476, 805104, 634175, 1250536, 805104])))
    model = fit_multi_label_mnb.fit(test_y, test_x)
    predict_res = predict_multi_label_mnb.predict(test_x, model)
    assert len(predict_res) == 3
    assert predict_res[0] == [76255]
    assert predict_res[1] == [76255]
    assert predict_res[2] == [76255]
