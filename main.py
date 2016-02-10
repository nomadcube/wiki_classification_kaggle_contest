import sys
from time import time

from read import Sample
from preprocessing.tf_idf import tf_idf
from preprocessing.transforming import pick_features, dimension_reduction, convert_y_to_csr
from models.mnb import MNB
from metrics import macro_precision_recall


def main(in_path, subset_cnt, threshold, mnb_alpha):
    smp = Sample()
    smp.read(in_path)
    test_smp = smp.extract_and_update(subset_cnt)

    tf_idf_x = tf_idf(smp.x)
    good_features = pick_features(tf_idf_x.indices, tf_idf_x.data, threshold)

    y_mat = convert_y_to_csr(smp.y, max_n_dim=smp.class_cnt)
    reduction_x = dimension_reduction(smp.x, good_features)

    test_reduction_x = dimension_reduction(test_smp.x, good_features)

    m = MNB(mnb_alpha)
    m.fit(y_mat, reduction_x)

    predicted_y = m.predict(reduction_x)
    test_predicted_y = m.predict(test_reduction_x)

    print predicted_y
    print test_predicted_y

    return macro_precision_recall(smp.y, predicted_y, smp.class_cnt), macro_precision_recall(test_smp.y,
                                                                                             test_predicted_y,
                                                                                             smp.class_cnt)


if __name__ == '__main__':
    in_p = sys.argv[1] if len(sys.argv) > 1 else '/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt'
    sc = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    t = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    alpha = int(sys.argv[4]) if len(sys.argv) > 4 else 0.

    start_time = time()
    print(main(in_p, sc, t, alpha))
    print(time() - start_time)
