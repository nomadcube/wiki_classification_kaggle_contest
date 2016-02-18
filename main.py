# coding=utf-8
import sys
from time import time

from read import Sample
from preprocessing.transforming import YConverter, XConverter, convert_y_to_csr
from models.mnb import MNB
from metrics import macro_precision_recall
from memory_profiler import profile
from preprocessing.tf_idf import tf_idf


# @profile
def main(in_path, tf_idf_threshold, mnb_smooth_coef, predict_cnt):
    smp = Sample()
    smp.read(in_path)
    train_smp, test_smp = smp.extract_and_update()
    print len(train_smp)
    print len(test_smp)

    x_converter = XConverter(tf_idf_threshold)
    x_converter.construct(train_smp.x)
    print len(x_converter.selected_features)

    mapped_reduced_x = tf_idf(x_converter.convert(train_smp.x))
    mapped_reduced_test_x = tf_idf(x_converter.convert(test_smp.x))

    y_converter = YConverter()
    y_converter.construct(smp.y)

    mapped_y = y_converter.convert(train_smp.y)

    mnb = MNB(mnb_smooth_coef)
    mnb.fit(convert_y_to_csr(mapped_y), mapped_reduced_x)
    mapped_test_predicted_y = mnb.predict(mapped_reduced_test_x, predict_cnt)

    mapped_test_y = y_converter.convert(test_smp.y)

    return macro_precision_recall(mapped_test_y, mapped_test_predicted_y)


if __name__ == '__main__':
    IN_PATH = sys.argv[1] if len(
        sys.argv) > 1 else '/Users/wumengling/PycharmProjects/kaggle/input_data/small_origin_train_subset.csv'
    TF_IDF_THRESHOLD = float(sys.argv[2]) if len(sys.argv) > 2 else 99.5
    MNB_SMOOTH_COEF = float(sys.argv[3]) if len(sys.argv) > 3 else 1.
    PREDICT_CNT = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    start_time = time()

    # import cProfile, pstats, StringIO
    #
    # pr = cProfile.Profile()
    # pr.enable()
    #
    print(main(IN_PATH, TF_IDF_THRESHOLD, MNB_SMOOTH_COEF, PREDICT_CNT))

    # pr.disable()
    # s = StringIO.StringIO()
    # sortby = 'tottime'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print s.getvalue()

    print(time() - start_time)
