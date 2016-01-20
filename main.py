import sys
from time import time

from reader import sample_reader

train_sample_path = sys.argv[1] if len(
        sys.argv) >= 2 else '/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv'
sample_size = int(sys.argv[2]) if len(sys.argv) >= 3 else 10

read_start_time = time()
sample = sample_reader(train_sample_path, sample_size)
print("time used for sample_reader is {0}".format(time() - read_start_time))
print(len(sample[0]))
print(len(sample[1]))
print(len(sample[2]))
print(len(sample[3]))

# tf_idf_start_time = time()
# x_tf_idf = tf_idf(sample.x)
# print("time used for tf_idf is {0}".format(time() - tf_idf_start_time))
#
# print(sample.x)
# print(x_tf_idf.shape)
# print(x_tf_idf.nnz)
