# coding=utf-8
import sys
from time import time

from read import Sample
from preprocessing.transforming import label_mapping, XConverter
from models.mnb import MNB
from models.lr import LR
from metrics import macro_precision_recall


def main(in_path, subset_cnt, threshold, mnb_alpha):
    smp = Sample()
    smp.read(in_path)
    test_smp = smp.extract_and_update(subset_cnt)

    x_converter = XConverter(threshold)
    x_converter.construct(smp.x)
    print len(x_converter.selected_features)

    mapped_reduced_x = x_converter.convert(smp.x)
    mapped_reduced_test_x = x_converter.convert(test_smp.x)

    mapped_y = label_mapping(smp.y)

    m = LR(0, 2)
    m.fit(mapped_y, mapped_reduced_x.todense())

    test_predicted_y = m.predict(mapped_reduced_test_x.todense())

    return macro_precision_recall(test_smp.y, test_predicted_y, smp.class_cnt)


if __name__ == '__main__':
    in_p = sys.argv[1] if len(
        sys.argv) > 1 else '/Users/wumengling/PycharmProjects/kaggle/input_data/small_origin_train_subset.csv'
    sc = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    t = float(sys.argv[3]) if len(sys.argv) > 3 else 97
    alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.

    start_time = time()

    import cProfile, pstats, StringIO

    pr = cProfile.Profile()
    pr.enable()

    print(main(in_p, sc, t, alpha))

    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()

    print(time() - start_time)
