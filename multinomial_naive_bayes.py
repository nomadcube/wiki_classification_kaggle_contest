import sys

from time import time
from data_processing.transformation import sample_reader, describe
from pympler.asizeof import asizeof

train_sample_path = sys.argv[1] if len(
        sys.argv) >= 2 else '/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv'
sample_size = sys.argv[2] if len(sys.argv) >= 3 else 100000

samples = sample_reader(train_sample_path, sample_size)
