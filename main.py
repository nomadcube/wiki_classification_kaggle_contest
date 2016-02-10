import sys
from time import time

from read import Sample
from preprocessing.tf_idf import tf_idf
from preprocessing.transforming import pick_features, dimension_reduction, convert_y_to_csr
from models.mnb import MNB
from metrics import macro_precision_recall


def main(in_path, threshold, mnb_alpha):
    smp = Sample()
    smp.read(in_path)

    tf_idf_x = tf_idf(smp.x)
    good_features = pick_features(tf_idf_x.indices, tf_idf_x.data, threshold)

    y_mat = convert_y_to_csr(smp.y, max_n_dim=smp.class_cnt)
    reduction_x = dimension_reduction(smp.x, good_features)

    m = MNB(mnb_alpha)
    m.fit(y_mat, reduction_x)

    predicted_y = m.predict(reduction_x)

    return macro_precision_recall(smp.y, predicted_y, smp.class_cnt)


if __name__ == '__main__':
    in_p = sys.argv[1] if len(sys.argv) > 1 else '/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt'
    t = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    alpha = int(sys.argv[3]) if len(sys.argv) > 3 else 0.

    start_time = time()
    print(main(in_p, t, alpha))
    print(time() - start_time)
