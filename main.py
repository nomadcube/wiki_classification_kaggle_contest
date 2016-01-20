import sys

from cs_tf_idf import tf_idf
from reader import sample_reader


train_sample_path = sys.argv[1] if len(
        sys.argv) >= 2 else '/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv'
sample_size = sys.argv[2] if len(sys.argv) >= 3 else 10

sample = sample_reader(train_sample_path, sample_size)
x_tf_idf = tf_idf(sample.x)
print(x_tf_idf.nnz)
