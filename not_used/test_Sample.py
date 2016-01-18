from data_processing.transformation.hierarchy import hierarchy

import data_processing.transformation


class TestSample:
    def pytest_funcarg__full_sample(self):
        return data_processing.transformation.base_sample_reader(
            '/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt')

    def pytest_funcarg__hierarchy(self):
        upwarded_hierarchy = hierarchy.HierarchyTable()
        upwarded_hierarchy.read_data('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/fake_hierarchy.txt')
        upwarded_hierarchy.update(0)
        return upwarded_hierarchy

    def test_train_initiation(self, full_sample):
        assert full_sample.y == {0: '314523,165538,416827',
                                 1: '21631',
                                 2: '76255,335416'}
        assert full_sample.x == {0: {1250536: 1},
                                 1: {634175: 1,
                                     1095476: 4,
                                     805104: 1},
                                 2: {1250536: 1,
                                     805104: 1}}

    def test_train_data_x_dim_reduction(self, full_sample):
        full_sample.dimension_reduction(-1.0)
        assert round(full_sample.x[0][1250536], 2) == 0.41
        assert round(full_sample.x[1][634175], 2) == 0.18
        assert round(full_sample.x[1][1095476], 2) == 0.73
        assert round(full_sample.x[1][805104], 2) == 0.07
        assert round(full_sample.x[2][1250536], 2) == 0.20
        assert round(full_sample.x[2][805104], 2) == 0.20

    def test_disassembled_label_upward(self, hierarchy, full_sample):
        full_sample.label_upward(hierarchy)
        assert len(full_sample.y) == 3
        assert full_sample.y == {0: '165538,314523',
                                 1: '416827',
                                 2: '21631,233333'}

    def test_convert_to_binary_class(self, full_sample):
        full_sample.convert_to_binary_class('76255')
        assert full_sample.binary_y == {0: -1,
                                        1: -1,
                                        2: 1}

    def test_split(self, full_sample):
        full_sample.convert_to_binary_class('76255')
        train_y, train_x, test_y, test_x = full_sample.split_train_test(0.5)[:4]
        assert len(train_y) == 1
        assert len(test_y) == 2
