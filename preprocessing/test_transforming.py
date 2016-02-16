from transforming import pick_features, dimension_reduction, feature_mapping
from tf_idf import tf_idf
import numpy as np
from scipy.sparse import csr_matrix


class TestTransformation:
    def pytest_funcarg__origin_x(self):
        element = np.array([1., 1., 1., 4., 1., 1.])
        row_index = np.array([0, 1, 1, 1, 2, 2])
        col_index = [1250536, 634175, 805104, 1095476, 805104, 1250536]
        return csr_matrix((element, (row_index, col_index)), shape=(max(row_index) + 1, max(col_index) + 1))

    def test_pick_features(self, origin_x):
        tf_idf_x = tf_idf(origin_x)
        features = pick_features(tf_idf_x.indices, tf_idf_x.data, 100)
        assert len(features) == 1
        assert 1095476 in features

    def pytest_funcarg__features(self, origin_x):
        tf_idf_x = tf_idf(origin_x)
        return pick_features(tf_idf_x.indices, tf_idf_x.data, 100)

    def test_dimension_reduction(self, origin_x, features):
        reduced_x = dimension_reduction(origin_x, features)
        assert isinstance(reduced_x, csr_matrix)
        assert reduced_x.shape == (3, 1095477)
        assert reduced_x.nnz == 1

    def pytest_funcarg__reduced_x(self, origin_x, features):
        return dimension_reduction(origin_x, features)

    def test_feature_mapping(self, reduced_x, features):
        mapped_x = feature_mapping(reduced_x, features)
        assert isinstance(mapped_x, csr_matrix)
        assert mapped_x.shape == (3, 1)
        assert mapped_x[1, 0] == 4.
