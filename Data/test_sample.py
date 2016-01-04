import Sample
from Data.hierarchy import hierarchy


class TestSample:
    def pytest_funcarg__TR(self):
        return Sample.sample_reader('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt')

    def pytest_funcarg__hierarchy_f_path(self):
        return '/Users/wumengling/PycharmProjects/kaggle/unit_test_data/fake_hierarchy.txt'

    def test_train_data_init(self, TR):
        assert TR.y == {0: '314523,165538,416827',
                        1: '21631',
                        2: '76255,335416'}
        assert TR.x == {0: {1250536: 1},
                        1: {634175: 1,
                            1095476: 4,
                            805104: 1},
                        2: {1250536: 1,
                            805104: 1}}

    def test_train_data_x_dim_reduction(self, TR):
        TR.dimension_reduction(-1.0)
        assert round(TR.x[0][1250536], 2) == 0.41

        assert round(TR.x[1][634175], 2) == 0.18
        assert round(TR.x[1][1095476], 2) == 0.73
        assert round(TR.x[1][805104], 2) == 0.07

        assert round(TR.x[2][1250536], 2) == 0.20
        assert round(TR.x[2][805104], 2) == 0.20

    def test_disassembled_label_upward(self, hierarchy_f_path, TR):
        upwarded_hierarchy = hierarchy.HierarchyTable()
        upwarded_hierarchy.read_data('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/fake_hierarchy.txt')
        upwarded_hierarchy.update(0)
        assert TR.y == {0: '314523,165538,416827',
                        1: '21631',
                        2: '76255,335416'}
        TR.label_upward(upwarded_hierarchy)
        assert len(TR.y) == 3
        assert TR.y == {0: '314523,165538,165538',
                        1: '416827',
                        2: '21631,233333'}

    def test_convert_to_binary_class(self, TR):
        TR.convert_to_binary_class('76255')
        assert TR.binary_y == {0: -1,
                               1: -1,
                               2: 1}

    def test_split(self, TR):
        TR.convert_to_binary_class('76255')
        train_y, train_x, test_y, test_x = TR.split_train_test(0.5)[:4]
        assert len(train_y) == 1
        assert len(test_y) == 2
