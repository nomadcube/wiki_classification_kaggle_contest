# coding=utf-8
import sys
from time import time

from read import Sample
from preprocessing.tf_idf import tf_idf
from preprocessing.transforming import pick_features, dimension_reduction, convert_y_to_csr, feature_mapping
from models.mnb import MNB
from models.lr import LR
from metrics import macro_precision_recall


def main(in_path, subset_cnt, threshold, mnb_alpha):
    smp = Sample()
    smp.read(in_path)
    test_smp = smp.extract_and_update(subset_cnt)

    tf_idf_x = tf_idf(smp.x)
    good_features = pick_features(tf_idf_x.indices, tf_idf_x.data, threshold)
    print len(good_features)

    reduction_x = dimension_reduction(smp.x, good_features)
    test_reduction_x = dimension_reduction(test_smp.x, good_features)

    mapped_reduced_x = feature_mapping(reduction_x, good_features)
    mapped_reduced_test_x = feature_mapping(test_reduction_x, good_features)

    m = LR(0, 2)
    m.fit(smp.y, mapped_reduced_x.todense())

    test_predicted_y = m.predict(mapped_reduced_test_x)

    return macro_precision_recall(test_smp.y, test_predicted_y, smp.class_cnt)


if __name__ == '__main__':
    in_p = sys.argv[1] if len(sys.argv) > 1 else '/Users/wumengling/PycharmProjects/kaggle/input_data/train_subset.csv'
    sc = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    t = float(sys.argv[3]) if len(sys.argv) > 3 else 98
    alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.

    start_time = time()
    print(main(in_p, sc, t, alpha))
    print(time() - start_time)
