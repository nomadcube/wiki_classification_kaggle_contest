import numpy as np
from sklearn import datasets

from learning import chunking
from learning.multinomial_naive_bayes_csr import construct_csr_from_array, MultinomialNB

iris = datasets.load_iris()


class TestMNB:
    def pytest_funcarg__y(self):
        return iris.target

    def pytest_funcarg__x(self):
        return iris.data

    def test_construct_csr_from_array(self, y, x):
        x_csr = construct_csr_from_array(x)
        nb = MultinomialNB()
        nb_classifier = nb.fit(x_csr, y)
        y_predict = nb_classifier.predict(x_csr)
        assert list(y_predict) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                   1, 1,
                                   1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1,
                                   1, 1,
                                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                   2, 2,
                                   2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                   2, 2]

    def test_out_of_core_learning(self, y, x):
        mnb = MultinomialNB()
        classes = [0, 1, 2]
        for chuck_y, chuck_x in chunking.iter_chuck(y.__iter__(), x.__iter__(), 15):
            csr_chuck_x = construct_csr_from_array(np.array(chuck_x))
            mnb.partial_fit(csr_chuck_x, chuck_y, classes=classes)
        y_predict = mnb.predict(construct_csr_from_array(x))
        assert list(y_predict) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                   1, 1,
                                   1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1,
                                   1, 1,
                                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                   2, 2,
                                   2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                   2, 2]
