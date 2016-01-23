import sys
from time import time

from scipy.sparse import csr_matrix

import multi_label_mnb
import reader
import sparse_tf_idf

sample_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt'
size_of_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 3

start_time = time()

# read data from file
smp = reader.read_sample(sample_path, size_of_sample)
smp_x = csr_matrix((smp.element_x, (smp.row_index_x, smp.col_index_x)),
                   shape=(max(smp.row_index_x) + 1, max(smp.col_index_x) + 1))

# perform tf-idf
tf_idf_smp_x = sparse_tf_idf.tf_idf(smp_x)

# fit mnb model
model = multi_label_mnb.fit(smp.y, tf_idf_smp_x)
print(model[0].shape)
print(model[1].shape)
print(model[1].nnz)
predict_sample_per_label = multi_label_mnb.predict(tf_idf_smp_x, model)
print(len(predict_sample_per_label))
print(time() - start_time)
