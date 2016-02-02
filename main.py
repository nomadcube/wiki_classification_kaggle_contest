import gc
import sys
from time import time

from guppy import hpy
from scipy.sparse import csr_matrix
from memory_profiler import profile

import evaluation
import fit_multi_label_mnb
import predict_multi_label_mnb
import reader


# @profile
def main(sample_path, size_of_sample, size_of_train_sample, predict_label_cnt):
    # read data from file
    train_smp, test_smp = reader.read_sample(sample_path, size_of_sample, size_of_train_sample)
    n_feature = max(train_smp.max_feature, test_smp.max_feature) + 1
    n_class_label = max(train_smp.max_class_label, test_smp.max_class_label) + 1
    train_x = csr_matrix((train_smp.element_x, train_smp.col_index_x, train_smp.row_indptr_x),
                         shape=(len(train_smp.row_indptr_x) - 1, n_feature), dtype='float')

    # fit non-smoothed mnb model
    m = fit_multi_label_mnb.fit(train_smp.y, train_x)

    # make prediction on test and train sample
    test_x = csr_matrix((test_smp.element_x, test_smp.col_index_x, test_smp.row_indptr_x),
                        shape=(len(test_smp.row_indptr_x) - 1, n_feature), dtype='float')
    test_predict = predict_multi_label_mnb.predict(test_x, m, 1)
    h = hpy()
    print h.heap()
    return evaluation.macro_precision_recall(test_smp.y, test_predict, n_class_label)


if __name__ == '__main__':
    import cProfile, pstats, StringIO

    # pr = cProfile.Profile()
    # pr.enable()
    #
    start_time = time()
    sample_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv'
    size_of_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    size_of_train_sample = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    predict_label_cnt_per_sample = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    print(main(sample_path, size_of_sample, size_of_train_sample, predict_label_cnt_per_sample))
    h = hpy()
    print h.heap()
    print(time() - start_time)

    # pr.disable()
    # s = StringIO.StringIO()
    # sortby = 'tottime'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print s.getvalue()
