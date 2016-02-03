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
def main(sample_path, size_train, size_test, predict_label_cnt):
    # read data from file
    test, train, part_train = reader.read_sample(sample_path, size_train, size_test)
    n_feature = max(train.max_feature, test.max_feature, part_train.max_feature) + 1
    n_class_label = max(train.max_class_label, test.max_class_label, part_train.max_feature) + 1

    # convert to csr matrix
    x_test = test.convert_to_csr(n_feature)
    x_train = train.convert_to_csr(n_feature)
    x_part_train = part_train.convert_to_csr(n_feature)

    # fit non-smoothed mnb model
    m = fit_multi_label_mnb.fit(train.y, x_train)

    # make prediction on test,  train and cv sample
    predict_part_train = predict_multi_label_mnb.predict(x_part_train, m, predict_label_cnt)
    predict_test = predict_multi_label_mnb.predict(x_test, m, predict_label_cnt)
    print len(predict_part_train)
    print len(predict_test)
    print evaluation.macro_precision_recall(test.y, predict_test, n_class_label)
    print evaluation.macro_precision_recall(part_train.y, predict_part_train, n_class_label)

    return 0


if __name__ == '__main__':
    import cProfile, pstats, StringIO

    # pr = cProfile.Profile()
    # pr.enable()

    start_time = time()

    sample_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv'
    size_train = int(sys.argv[2]) if len(sys.argv) > 3 else 1000
    size_test = int(sys.argv[3]) if len(sys.argv) > 2 else 100
    cnt_predict_class = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    print(main(sample_path, size_train, size_test, cnt_predict_class))
    print(time() - start_time)

    # pr.disable()
    # s = StringIO.StringIO()
    # sortby = 'tottime'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print s.getvalue()
