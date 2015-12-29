import Sample


class TestSample:
    def pytest_funcarg__TR(self):
        return Sample.sample_reader('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt')

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

    def test_train_data_y_remapped(self, TR):
        TR.label_string_remap()
        assert TR.y_remapping_rel == {'314523,165538,416827': 0,
                                      '21631': 1,
                                      '76255,335416': 2}
        assert TR.y == {0: 0,
                        1: 1,
                        2: 2}

    def test_train_data_x_dim_reduction(self, TR):
        TR.dimension_reduction(-1.0)
        assert round(TR.x[0][1250536], 2) == 0.41

        assert round(TR.x[1][634175], 2) == 0.18
        assert round(TR.x[1][1095476], 2) == 0.73
        assert round(TR.x[1][805104], 2) == 0.07

        assert round(TR.x[2][1250536], 2) == 0.20
        assert round(TR.x[2][805104], 2) == 0.20

    def test_split(self, TR):
        train_y, train_x, test_y, test_x = TR.label_string_remap().dimension_reduction(-1.0).split_train_test(0.4)
        assert len(train_y) == 1
        assert len(test_y) == 2

    def test_convert_to_binary_class(self, TR):
        TR.convert_to_binary_class('76255')
        assert TR.y == {0: -1,
                        1: -1,
                        2: 1}

    def test_label_string_disassemble(self, TR):
        TR.label_string_disassemble()
        assert len(TR.y) == 6
        assert len(TR.x) == 6
        assert TR.y == {0: '314523',
                        1: '165538',
                        2: '416827',
                        3: '21631',
                        4: '76255',
                        5: '335416'}
        assert TR.x == {0: {1250536: 1},
                        1: {1250536: 1},
                        2: {1250536: 1},
                        3: {634175: 1,
                            1095476: 4,
                            805104: 1},
                        4: {1250536: 1,
                            805104: 1},
                        5: {1250536: 1,
                            805104: 1}}
