# coding=utf-8
from itertools import product
from read import Sample
from preprocessing.transforming import YConverter, XConverter, convert_y_to_csr
from metrics import macro_precision_recall
from preprocessing.tf_idf import tf_idf
from memory_profiler import profile
import sys


class PipeLine:
    def __init__(self, model_class, threshold, predict_cnt):
        self._model = model_class
        self._threshold = threshold
        self._predict_cnt = predict_cnt

        self.best_f_score = 0.
        self.best_x_converter = None
        self.best_y_converter = None
        self.best_model = None
        self.best_threshold = None
        self.best_predicted_cnt = None

    @profile
    def run(self, in_path):
        smp = Sample()
        smp.read(in_path)
        train_smp, test_smp, common_labels_cnt = smp.extract_and_update()

        y_converter = YConverter()
        y_converter.construct(smp.label_old_new_relation)
        mapped_y = y_converter.convert(train_smp.y)

        for param in product(self._threshold, self._predict_cnt):
            tf_idf_threshold, predict_cnt = param

            x_converter = XConverter(tf_idf_threshold)
            x_converter.construct(train_smp.x)

            mapped_reduced_x = tf_idf(x_converter.convert(train_smp.x))
            mapped_reduced_test_x = tf_idf(x_converter.convert(test_smp.x))

            mnb = self._model()
            mnb.fit(convert_y_to_csr(mapped_y), mapped_reduced_x)

            mapped_test_predicted_y = mnb.predict(mapped_reduced_test_x, predict_cnt)
            mapped_test_y = y_converter.convert(test_smp.y)

            mpr_mre = macro_precision_recall(mapped_test_y, mapped_test_predicted_y,
                                             len(y_converter.label_old_new_relation), common_labels_cnt)
            f_score = 1. / (1. / mpr_mre[0] + 1. / mpr_mre[1]) if mpr_mre[0] != 0. and mpr_mre[1] != 0. else float(
                "inf")
            print mpr_mre
            print f_score

            if f_score > self.best_f_score:
                self.best_f_score = round(f_score, 3)
                self.best_model = mnb
                self.best_x_converter = x_converter
                self.best_y_converter = y_converter
                self.best_threshold = tf_idf_threshold
                self.best_predicted_cnt = predict_cnt

    def __repr__(self):
        return "best_threshold: {0}\nbest_predicted_cnt: {1}".format(self.best_threshold, self.best_predicted_cnt)


if __name__ == '__main__':
    from models.mnb import LaplaceSmoothedMNB

    PATH = sys.argv[1] if len(
        sys.argv) > 1 else '/Users/wumengling/PycharmProjects/kaggle/input_data/small_origin_train_subset.csv'
    cv = PipeLine(LaplaceSmoothedMNB, [90], [1])
    cv.run(PATH)
    print cv.best_predicted_cnt
    print cv.best_f_score
    print cv.best_model
