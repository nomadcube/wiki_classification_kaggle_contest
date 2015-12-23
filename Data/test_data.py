import Data


class TestTrainData:
    def pytest_funcarg__TR(self):
        return Data.Data('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt')

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
        TR.remap()
        assert TR.y_remapping_rel == {'314523,165538,416827': 0,
                                      '21631': 1,
                                      '76255,335416': 2}
        assert TR.y == {0: 0,
                        1: 1,
                        2: 2}

    def test_train_data_x_dim_reduction(self, TR):
        TR.dim_reduction(-1.0)
        assert round(TR.x[0][1250536], 2) == 0.41

        assert round(TR.x[1][634175], 2) == 0.18
        assert round(TR.x[1][1095476], 2) == 0.73
        assert round(TR.x[1][805104], 2) == 0.07

        assert round(TR.x[2][1250536], 2) == 0.20
        assert round(TR.x[2][805104], 2) == 0.20

    def test_split(self, TR):
        TR.remap().dim_reduction(-1.0).sample_split(1)
        assert len(TR.train_y) == 1
