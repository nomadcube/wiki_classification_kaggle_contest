# coding=utf-8
from itertools import product
from read import Sample
from preprocessing.transforming import YConverter, XConverter, convert_y_to_csr
from models.mnb import MNB
from metrics import macro_precision_recall
from preprocessing.tf_idf import tf_idf
import sys


class PipeLine:
    def __init__(self, threshold, smooth_coef, predict_cnt):
        self._threshold = threshold
        self._smooth_coef = smooth_coef
        self._predict_cnt = predict_cnt
        self.pipeline_info = dict()
        self.x_converter = None
        self.y_converter = None
        self.best_model = None
        self.best_f_score = 0.

    def run(self, in_path):
        smp = Sample()
        smp.read(in_path)
        train_smp, test_smp = smp.extract_and_update()

        self.pipeline_info['train_size'] = len(train_smp)
        self.pipeline_info['test_size'] = len(test_smp)

        y_converter = YConverter()
        y_converter.construct(smp.y)
        mapped_y = y_converter.convert(train_smp.y)

        for param in product(self._threshold, self._smooth_coef, self._predict_cnt):
            tf_idf_threshold, mnb_smooth_coef, predict_cnt = param
            experiment_no = "threshold: {0}, smooth: {1}, predict_cnt: {2}".format(tf_idf_threshold, mnb_smooth_coef,
                                                                                   predict_cnt)
            self.pipeline_info[experiment_no] = dict()

            x_converter = XConverter(tf_idf_threshold)
            x_converter.construct(train_smp.x)
            self.pipeline_info[experiment_no]['feature_cnt'] = len(x_converter.selected_features)

            mapped_reduced_x = tf_idf(x_converter.convert(train_smp.x))
            mapped_reduced_test_x = tf_idf(x_converter.convert(test_smp.x))

            mnb = MNB(mnb_smooth_coef)
            mnb.fit(convert_y_to_csr(mapped_y), mapped_reduced_x)

            mapped_test_predicted_y = mnb.predict(mapped_reduced_test_x, predict_cnt)
            mapped_test_y = y_converter.convert(test_smp.y)

            mpr_mre = macro_precision_recall(mapped_test_y, mapped_test_predicted_y)
            f_score = 1. / (1. / mpr_mre[0] + 1. / mpr_mre[1])

            self.pipeline_info[experiment_no]['f_score'] = round(f_score, 3)

            if f_score > self.best_f_score:
                self.best_f_score = f_score
                self.best_model = mnb
                self.x_converter = x_converter
                self.y_converter = y_converter


if __name__ == '__main__':
    PATH = sys.argv[1] if len(
        sys.argv) > 1 else '/Users/wumengling/PycharmProjects/kaggle/input_data/small_origin_train_subset.csv'
    cv = PipeLine([97, 95, 93], [1.0, 0.0], [1, 2, 3])
    cv.run(PATH)
    print cv.pipeline_info
    print cv.best_f_score
    print cv.best_model
