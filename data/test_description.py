import description


class TestDescription:
    def pytest_funcarg__x(self):
        return [{'1250536': 1},
                {'1250536': 1},
                {'1250536': 1},
                {'634175': 1, '1095476': 4, '805104': 1},
                {'1250536': 1, '805104': 1},
                {'1250536': 1, '805104': 1}]

    def pytest_funcarg__y(self):
        return ['314523',
                '165538',
                '416827',
                '21631',
                '76255',
                '335416']

    def test_collect_feature(self, x):
        feature_res = description.collect_feature(x)
        assert len(feature_res) == 4

    def test_each_class_count(self, y):
        class_count_res = description.each_class_count(y)
        assert len(class_count_res) == 6
        assert class_count_res['314523'] == 1
        assert class_count_res['165538'] == 1
        assert class_count_res['416827'] == 1
        assert class_count_res['21631'] == 1
        assert class_count_res['76255'] == 1
        assert class_count_res['335416'] == 1

    def test_describe_x_y(self, x, y):
        desc = description.describe_x_y(x, y)
        assert desc.sample_size == 6
        assert desc.feature_dimension == 4
        assert desc.class_distribution['314523'] == 1
        assert desc.class_distribution['165538'] == 1
        assert desc.class_distribution['416827'] == 1
        assert desc.class_distribution['21631'] == 1
        assert desc.class_distribution['76255'] == 1
        assert desc.class_distribution['335416'] == 1
