from perceptron import if_wrong_discriminated, update_weight, learning, predict
from sparse_vector.sparse_vector import sparse_vector_dat_t, SparseVector


class TestPerceptron:
    def pytest_funcarg__x(self):
        return SparseVector(sparse_vector_dat_t({1: 3.0, 2: 3.0}))

    def pytest_funcarg__y(self):
        return 1.0

    def pytest_funcarg__w(self):
        return SparseVector(sparse_vector_dat_t({1: 0.0, 2: 0.0}))

    def test_if_wrong_discriminated(self, x, y, w):
        res = if_wrong_discriminated(x, y, w)
        assert res is True

    def pytest_funcarg__neg_gradient(self):
        return SparseVector(sparse_vector_dat_t({1: 3.0, 2: 3.0}))

    def test_update_weight(self, w, neg_gradient):
        res = update_weight(w, 1, neg_gradient)
        assert dict(res.dat) == {1: 3.0,
                                 2: 3.0}

    def pytest_funcarg__X(self):
        return [SparseVector(sparse_vector_dat_t({1: 3.0, 2: 3.0, 3: 1.0})),
                SparseVector(sparse_vector_dat_t({1: 4.0, 2: 3.0, 3: 1.0})),
                SparseVector(sparse_vector_dat_t({1: 1.0, 2: 1.0, 3: 1.0}))]

    def pytest_funcarg__Y(self):
        return [1.0, 1.0, -1.0]

    def test_perceptron(self, X, Y):
        final_w = learning(100, 1.0, X, Y, 3)
        assert dict(final_w.dat) == {1: 1.0,
                                     2: 1.0,
                                     3: -3.0}

    def pytest_funcarg__fitted_w(self):
        return SparseVector(sparse_vector_dat_t({1: 1.0,
                                                 2: 1.0,
                                                 3: -3.0}))

    def test_predict(self, x, fitted_w):
        predict_res = predict(fitted_w, x)
        assert predict_res == 1.0
