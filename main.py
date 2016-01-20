import math
import sys
from time import time

import reader_and_serializer
import sparse_tf_idf

sample_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt'
size_of_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 3
part_sample_path = sys.argv[3] if len(sys.argv) > 3 else '/Users/wumengling/PycharmProjects/kaggle/part_sample'

start_time = time()
sample_1, sample_2 = reader_and_serializer.read_part_sample(sample_path, size_of_sample,
                                                            math.floor(size_of_sample / 2.0))
column = sample_1.col_index_x + sample_2.col_index_x
total_idf = sparse_tf_idf.idf(column, size_of_sample)

reader_and_serializer.save_part_sample([sample_1, sample_2], part_sample_path)

for each_tf_idf in sparse_tf_idf.part_tf_idf_generator(part_sample_path, total_idf):
    print(each_tf_idf.shape)
print(start_time - time())
