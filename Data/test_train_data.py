from Data.TrainData import TrainData
import math


class TestTrainData:
    def pytest_funcarg__TR(self):
        return TrainData('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt')

    def test_train_data_init(self, TR):
        assert TR.y.data == {0: '314523,165538,416827',
                             1: '21631',
                             2: '76255,335416'}
        assert TR.x.data == {0: {1250536: 1},
                             1: {634175: 1,
                                 1095476: 4,
                                 805104: 1},
                             2: {1250536: 1,
                                 805104: 1}}

    def test_train_data_y_remapped(self, TR):
        assert TR.y.remapping_relation == {'314523,165538,416827': 0,
                                            '21631': 1,
                                            '76255,335416': 2}
        assert TR.y.remapped_data == {0: 0,
                                      1: 1,
                                      2: 2}

    def test_train_data_x_dim_reduction(self, TR):
        TR.x.dim_reduction(0.0)
        assert round(TR.x.dim_reduction_data[0][1250536], 2) == 0.41

        assert round(TR.x.dim_reduction_data[1][634175], 2) == 0.18
        assert round(TR.x.dim_reduction_data[1][1095476], 2) == 0.73
        assert round(TR.x.dim_reduction_data[1][805104], 2) == 0.07

        assert round(TR.x.dim_reduction_data[2][1250536], 2) == 0.20
        assert round(TR.x.dim_reduction_data[2][805104], 2) == 0.20
