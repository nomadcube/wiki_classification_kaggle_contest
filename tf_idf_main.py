import os
import pickle

from sparse_tf_idf import idf, part_tf_idf_generator

output_dir = '/Users/wumengling/PycharmProjects/kaggle/output'
part_sample_dir = '/Users/wumengling/PycharmProjects/kaggle/output/part_sample'

if __name__ == '__main__':
    with open(os.path.join(output_dir, 'all_sample_size.obj'), 'r') as all_sample_size_f:
        all_sample_size = pickle.load(all_sample_size_f)
    with open(os.path.join(output_dir, 'all_column.obj'), 'r') as all_column_f:
        all_column = pickle.load(all_column_f)
    all_idf = idf(all_column, all_sample_size)
    print(all_idf)
    for t in part_tf_idf_generator(part_sample_dir, all_idf):
        print(t)
