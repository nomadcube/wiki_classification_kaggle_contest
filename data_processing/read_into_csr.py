from random import random
from scipy.sparse import csr_matrix
from pympler import asizeof
import numpy as np


def read_into_csr(data_file_path, sample_prop=1.0):
    element = list()
    row_index = list()
    col_index = list()
    row = 0
    with open(data_file_path, 'r') as f_stream:
        line = f_stream.readline()
        while line:
            determine_number = random()
            if determine_number <= sample_prop:
                tmp_x = line.strip().split(' ', 1)[1]
                for column in tmp_x.split(' '):
                    feature, val = column.split(':')
                    element.append(float(val))
                    row_index.append(int(row))
                    col_index.append(int(feature))
            row += 1
            line = f_stream.readline()
    return csr_matrix((element, (row_index, col_index)), shape=(max(row_index) + 1, max(col_index) + 1), dtype='int')


if __name__ == '__main__':
    sample_f_path = '/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv'
    csr_res = read_into_csr(sample_f_path, 0.0001)
    print(csr_res.shape)
    print(asizeof.asizeof(csr_res))
