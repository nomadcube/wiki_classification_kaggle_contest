import sys
from time import time

from cs_tf_idf import tf_idf
from reader import sample_reader


train_sample_path = sys.argv[1] if len(
        sys.argv) >= 2 else '/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv'
sample_size = int(sys.argv[2]) if len(sys.argv) >= 3 else 10

read_start_time = time()
sample = sample_reader(train_sample_path, sample_size)
print("time used for sample_reader is {0}".format(time() - read_start_time))

tf_idf_start_time = time()
x_tf_idf = tf_idf(sample.x)
print("time used for tf_idf is {0}".format(time() - tf_idf_start_time))

print(x_tf_idf.shape)
print(x_tf_idf.nnz)
