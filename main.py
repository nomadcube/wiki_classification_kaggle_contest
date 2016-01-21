import sys
from time import time

from scipy.sparse import csr_matrix

import reader
import sparse_tf_idf

sample_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt'
size_of_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 3

start_time = time()
smp = reader.read_sample(sample_path, size_of_sample)
smp_x = csr_matrix((smp.element_x, (smp.row_index_x, smp.col_index_x)),
                   shape=(max(smp.row_index_x) + 1, max(smp.col_index_x) + 1))
tf_idf_smp_x = sparse_tf_idf.tf_idf(smp_x)
print(tf_idf_smp_x.shape)
print(len(smp.y))
print(time() - start_time)
