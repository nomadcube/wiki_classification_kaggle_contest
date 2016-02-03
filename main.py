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
    test, train, cv = reader.read_sample(sample_path, size_of_sample, size_of_train_sample)
    n_feature = max(train.max_feature, test.max_feature, cv.max_feature) + 1
    n_class_label = max(train.max_class_label, test.max_class_label, cv.max_feature) + 1

    # convert to csr matrix
    x_test = test.convert_to_csr(n_feature)
    x_cv = cv.convert_to_csr(n_feature)
    x_train = train.convert_to_csr(n_feature)

    # fit non-smoothed mnb model
    m = fit_multi_label_mnb.fit(train.y, x_train)

    # make prediction on test,  train and cv sample
    predict_test = predict_multi_label_mnb.predict(x_test, m, predict_label_cnt)
    predict_train = predict_multi_label_mnb.predict(x_train, m, predict_label_cnt)
    predict_cv = predict_multi_label_mnb.predict(x_cv, m, predict_label_cnt)

    return evaluation.macro_precision_recall(train.y, predict_train, n_class_label), evaluation.macro_precision_recall(
        test.y, predict_test, n_class_label), evaluation.macro_precision_recall(cv.y, predict_cv, n_class_label)


if __name__ == '__main__':
    import cProfile, pstats, StringIO

    # pr = cProfile.Profile()
    # pr.enable()
    #
    start_time = time()

    sample_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv'
    size_test_or_cv = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    size_train = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    cnt_predict_class = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    print(main(sample_path, size_test_or_cv, size_train, cnt_predict_class))
    print(time() - start_time)

    # pr.disable()
    # s = StringIO.StringIO()
    # sortby = 'tottime'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print s.getvalue()
