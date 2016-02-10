# coding=utf-8
import sys
from time import time

import metrics
import preprocessing
import sparse_tf_idf
from mnb import fit, predict
from read import Sample


# @profile
def main(in_path):
    smp = Sample()
    smp.read(in_path)

    # perform feature selection
    s = read.read_sample(in_path, size_train + size_test, 0)[1]
    tfidf_x_train = sparse_tf_idf.tf_idf(s._convert_x_to_csr(s.max_feature + 1))
    good_features = preprocessing.high_tf_idf_features(tfidf_x_train, 99)
    print len(good_features)
    fm = {f: i for i, f in enumerate(good_features)}
    reduction_x_train = preprocessing.construct_lower_rank_x(x_train, good_features, fm)

    # fit non-smoothed mnb model
    m = fit.fit(train.y, reduction_x_train.tocsr())

    # make prediction on test,  train and cv sample
    reduction_x_part_train = preprocessing.construct_lower_rank_x(x_part_train, good_features, fm)
    reduction_x_test = preprocessing.construct_lower_rank_x(x_test, good_features, fm)

    predict_part_train = predict.predict(reduction_x_part_train, m)
    predict_test = predict.predict(reduction_x_test, m)

    return metrics.macro_precision_recall(test.y, predict_test, n_class_label), metrics.macro_precision_recall(
        part_train.y, predict_part_train, n_class_label)


if __name__ == '__main__':
    # import cProfile, pstats, StringIO
    #
    # pr = cProfile.Profile()
    # pr.enable()

    start_time = time()

    sample_path = sys.argv[1] if len(
        sys.argv) > 1 else '/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv'
    size_train = int(sys.argv[2]) if len(sys.argv) > 3 else 126
    size_test = int(sys.argv[3]) if len(sys.argv) > 2 else 10
    cnt_predict_class = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    print(main(sample_path, size_train, size_test, cnt_predict_class))
    print(time() - start_time)

    # pr.disable()
    # s = StringIO.StringIO()
    # sortby = 'tottime'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print s.getvalue()
