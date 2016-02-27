import numpy as np

from scipy.sparse import csr_matrix

from models.mnb import LaplaceSmoothedMNB
from unit_test.test_config import TestBase


class TestMnb(TestBase):
    def pytest_funcarg__model(self):
        return LaplaceSmoothedMNB('/Users/wumengling/PycharmProjects/kaggle/unit_test')

    def test_split_2(self, model, csr_y):
        all_part_y = [y for y in model.split(csr_y, 1, 2)]
        assert len(all_part_y) == 2
        assert all_part_y[0][1] == [0]
        assert all_part_y[1][1] == [1]

        assert all_part_y[0][0].nnz == 5
        assert all_part_y[0][0][0, 1] == 1.
        assert all_part_y[0][0][0, 4] == 1.
        assert all_part_y[0][0][0, 5] == 1.
        assert all_part_y[0][0][0, 6] == 1.
        assert all_part_y[0][0][0, 8] == 1.

        assert all_part_y[1][0].nnz == 5
        assert all_part_y[1][0][0, 0] == 1.
        assert all_part_y[1][0][0, 2] == 1.
        assert all_part_y[1][0][0, 3] == 1.
        assert all_part_y[1][0][0, 7] == 1.
        assert all_part_y[1][0][0, 9] == 1.
