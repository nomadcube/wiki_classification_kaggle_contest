import sys
from time import time

from scipy.sparse import csr_matrix

import multi_label_mnb
import reader
import sparse_tf_idf

sample_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv'
size_of_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 10
size_of_train_sample = int(sys.argv[3]) if len(sys.argv) > 3 else 5

start_time = time()

# read data from file
print(repr(time()) + ': read data starting')
train_smp, test_smp = reader.read_sample(sample_path, size_of_sample, size_of_train_sample)
n_dim = max(max(train_smp.col_index_x), max(test_smp.col_index_x)) + 1
train_x = csr_matrix((train_smp.element_x, (train_smp.row_index_x, train_smp.col_index_x)),
                     shape=(max(train_smp.row_index_x) + 1, n_dim))
test_x = csr_matrix((test_smp.element_x, (test_smp.row_index_x, test_smp.col_index_x)),
                    shape=(max(test_smp.row_index_x) + 1, n_dim))
print('read data end')

# perform tf-idf
print(repr(time()) + ': tf-idf starting')
train_tf_idf_x = sparse_tf_idf.tf_idf(train_x)
test_tf_idf_x = sparse_tf_idf.tf_idf(test_x)
print('tf-idf end')

# fit mnb model
print(repr(time()) + ': fit model starting')
model = multi_label_mnb.fit(train_smp.y, train_tf_idf_x)
print('fit model end')

# make prediction on test sample
print('make prediction starting')
predict_sample_per_label = multi_label_mnb.predict(test_tf_idf_x, model)
print('make prediction end')
print(len(predict_sample_per_label))
print(predict_sample_per_label.count(-1))
print(time() - start_time)
