import y_description, x_description


class TestDescription:
    def pytest_funcarg__y_dict(self):
        return {0: '314523,165538,416827',
                1: '21631',
                2: '76255,335416'}

    def pytest_funcarg__x_dict(self):
        return {0: {1250536: 1},
                1: {634175: 1, 1095476: 4, 805104: 1},
                2: {1250536: 1, 805104: 1}}

    def test_y_distribution(self, y_dict):
        res = y_description.y_distribution(y_dict)
        assert len(res) == 3
        assert res == {'314523,165538,416827': 1.0,
                       '21631': 1.0,
                       '76255,335416': 1.0}

    def test_descriptive_analysis(self):
        test_dat_list = [3, 4, 19, 1]
        res = y_description.descriptive_analysis(test_dat_list)
        assert res.min_val == 1
        assert res.max_val == 19
        assert res.median == 3
        assert res.mean_val == 27.0 / 4.0

    def test_x_description(self, x_dict):
        res = x_description.x_feature_set(x_dict)
        assert len(res) == 4
        assert (1250536 in res) and \
               (634175 in res) and \
               (1095476 in res) and \
               (805104 in res)
