import read
from preprocessing.transforming import convert_y_to_csr


class TestBase:
    def pytest_funcarg__infile(self):
        return '/Users/wumengling/PycharmProjects/kaggle/input_data/train_subset.csv'

    def pytest_funcarg__smp(self, infile):
        smp = read.Sample()
        smp.read(infile)
        return smp

    def pytest_funcarg__csr_y(self, smp):
        return convert_y_to_csr(smp.y)
