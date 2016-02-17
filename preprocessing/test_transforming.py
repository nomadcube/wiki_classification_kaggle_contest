from transforming import XConverter, label_mapping
import numpy as np
from scipy.sparse import csr_matrix


class TestTransformation:
    def pytest_funcarg__origin_x(self):
        element = np.array([1., 1., 1., 4., 1., 1.])
        row_index = np.array([0, 1, 1, 1, 2, 2])
        col_index = [1250536, 634175, 805104, 1095476, 805104, 1250536]
        return csr_matrix((element, (row_index, col_index)), shape=(max(row_index) + 1, max(col_index) + 1))

    def test_construct(self, origin_x):
        xc = XConverter(100)
        xc.construct(origin_x)
        assert len(xc.selected_features) == 1
        assert 1095476 in xc.selected_features

    def test_convert(self, origin_x):
        xc = XConverter(100)
        xc.construct(origin_x)
        mapped_x = xc.convert(origin_x)
        assert isinstance(mapped_x, csr_matrix)
        assert mapped_x.shape == (3, 1)
        assert mapped_x[1, 0] == 4.

    def pytest_funcarg__y(self):
        return [[314523, 165538, 416827], [21631], [76255, 165538]]

    def test_label_mapping(self, y):
        from array import array
        new_y = label_mapping(y)
        assert len(new_y) == 3
        assert new_y[0] == array('I', [0L, 1L, 2L])
        assert new_y[1] == array('I', [3L])
        assert new_y[2] == array('I', [4L, 1L])
