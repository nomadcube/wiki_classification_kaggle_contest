import sys

from time import time
from data_processing.transformation import sample_reader, describe
from pympler.asizeof import asizeof

path_train_sample = sys.argv[1] if len(
        sys.argv) >= 2 else '/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv'

max_line = 2365436
sample_size = 1000000

start_time = time()
samples = sample_reader(path_train_sample, sample_size, 100611105)
print(time() - start_time)
print(type(samples.y))
print(len(samples.y))
print(asizeof(samples.x))
print(sys.getsizeof(samples.x))
print(samples.x.shape)
print(samples.x.nnz)
