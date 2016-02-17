# coding=utf-8
import sys
from time import time

from read import Sample
from preprocessing.transforming import YConverter, XConverter
from models.mnb import MNB
from models.lr import LR
from metrics import macro_precision_recall
from memory_profiler import profile


# @profile
def main(in_path, threshold):
    smp = Sample()
    smp.read(in_path)
    smp, test_smp = smp.extract_and_update()
    print len(smp)
    print len(test_smp)

    x_converter = XConverter(threshold)
    x_converter.construct(smp.x)
    print len(x_converter.selected_features)

    mapped_reduced_x = x_converter.convert(smp.x)
    mapped_reduced_test_x = x_converter.convert(test_smp.x)

    y_converter = YConverter()
    y_converter.construct(smp.y)

    mapped_y = y_converter.convert(smp.y)

    m = LR(0, 2)
    m.fit(mapped_y, mapped_reduced_x.todense())

    test_predicted_y = m.predict(mapped_reduced_test_x.todense())
    old_test_predicted_y = y_converter.withdraw_convert(test_predicted_y)
    return macro_precision_recall(test_smp.y, old_test_predicted_y, smp.class_cnt)


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
