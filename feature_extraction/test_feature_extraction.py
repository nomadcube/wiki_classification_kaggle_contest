import feature_extraction


class TestFeatureExtraction:
    def pytest_funcarg__x_dict(self):
        return {0: {1250536: 0.4054651081081644,
                    1095476: 0.7324081924454064,
                    805104: 0.06757751801802739},
                1: {805104: 0.06757751801802739},
                2: {1250536: 0.2027325540540822,
                    805104: 0.2027325540540822}
                }

    def pytest_funcarg__subset(self):
        return {1250536, 805104}

    def test_feature_extraction(self, x_dict):
        res = feature_extraction.extraction(x_dict, threshold=0.1)
        assert res == {0: {1250536: 0.4054651081081644,
                           1095476: 0.7324081924454064},
                       1: {},
                       2: {1250536: 0.2027325540540822,
                           805104: 0.2027325540540822}
                       }

    def test_feature_extraction_with_subet(self, x_dict, subset):
        res = feature_extraction.extraction(x_dict, x_subset=subset)
        assert res == {0: {1250536: 0.4054651081081644,
                           805104: 0.06757751801802739},
                       1: {805104: 0.06757751801802739},
                       2: {1250536: 0.2027325540540822,
                           805104: 0.2027325540540822}
                       }
