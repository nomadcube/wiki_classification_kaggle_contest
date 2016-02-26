import read


class BaseTest:
    def pytest_funcarg__train_infile(self):
        return '/Users/wumengling/PycharmProjects/kaggle/input_data/train_subset.csv'
