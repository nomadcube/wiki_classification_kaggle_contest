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
def main(in_path, threshold):
    smp = Sample()
    smp.read(in_path)
    train_smp, test_smp = smp.extract_and_update()
    print len(train_smp)
    print len(test_smp)

    x_converter = XConverter(threshold)
    x_converter.construct(train_smp.x)
    print len(x_converter.selected_features)

    mapped_reduced_x = tf_idf(x_converter.convert(train_smp.x))
    mapped_reduced_test_x = tf_idf(x_converter.convert(test_smp.x))

    y_converter = YConverter()
    y_converter.construct(smp.y)

    mapped_y = y_converter.convert(train_smp.y)

    mnb = MNB(1.)
    mnb.fit(convert_y_to_csr(mapped_y), mapped_reduced_x)
    mapped_test_predicted_y = mnb.predict(mapped_reduced_test_x)

    mapped_test_y = y_converter.convert(test_smp.y)

    return macro_precision_recall(mapped_test_y, mapped_test_predicted_y)


if __name__ == '__main__':
    in_p = sys.argv[1] if len(
        sys.argv) > 1 else '/Users/wumengling/PycharmProjects/kaggle/input_data/small_origin_train_subset.csv'
    t = float(sys.argv[2]) if len(sys.argv) > 2 else 99.5

    start_time = time()

    # import cProfile, pstats, StringIO
    #
    # pr = cProfile.Profile()
    # pr.enable()
    #
    print(main(in_p, t))

    # pr.disable()
    # s = StringIO.StringIO()
    # sortby = 'tottime'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print s.getvalue()

    print(time() - start_time)
