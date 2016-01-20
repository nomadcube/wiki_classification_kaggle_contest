import math
import sys
from time import time

from reader import sample_reader

train_sample_path = sys.argv[1] if len(
        sys.argv) >= 2 else '/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv'
sample_size = int(sys.argv[2]) if len(sys.argv) >= 3 else 5

read_start_time = time()
sample_1, sample_2 = sample_reader(train_sample_path, 0, math.floor(sample_size / 2.0), sample_size)
print("time used for sample_reader is {0}".format(time() - read_start_time))
print(len(sample_1.y))
print(len(sample_2.y))
