# coding=utf-8
from itertools import product
from read import Sample
from preprocessing.transforming import YConverter, XConverter, convert_y_to_csr
from metrics import macro_precision_recall
from preprocessing.tf_idf import tf_idf
from memory_profiler import profile
import sys
import math
import pickle


class PipeLine:
    def __init__(self, model_type, threshold, num_predict, model_save_dir, submission_infile_save_dir, max_label_size):
        self._model = model_type
        self._threshold = threshold
        self._predict_cnt = num_predict

        self.best_f_score = 0.
        self.best_x_converter = None
        self.best_y_converter = None
        self.best_threshold = None
        self.best_predicted_cnt = None
        self.best_model = None

        self.model_store_dir = model_save_dir
        self.test_data_store_dir = submission_infile_save_dir
        self.max_label_size = max_label_size

    # @profile
    def model_selection(self, in_path, part_size):
        smp = Sample()
        smp.read(in_path)
        train_smp, test_smp, common_labels_cnt = smp.extract_and_update()

        y_converter = YConverter()
        y_converter.construct(train_smp.y)
        mapped_y = y_converter.convert(train_smp.y)
        # 26019898 function calls in 251.062 seconds

        for param in product(self._threshold, self._predict_cnt):
            tf_idf_threshold, predict_cnt = param

            x_converter = XConverter(tf_idf_threshold)
            x_converter.construct(smp.x)

            mapped_reduced_x = x_converter.convert(train_smp.x)
            mapped_reduced_test_x = x_converter.convert(test_smp.x)

            csr_mapped_y = convert_y_to_csr(mapped_y)
            print "num of labels in train set: {0}\n" \
                  "\ntrain set size: {1}\ntest set size:{3}" \
                  "\nfeature count: {2}\n".format(csr_mapped_y.shape[0], csr_mapped_y.shape[1],
                                                                  mapped_reduced_x.shape[1],
                                                                  mapped_reduced_test_x.shape[0])

            print "max size of y is {2}\nall y split into {0} parts, each with at most {1} labels\n".format(
                int(math.ceil(min(csr_mapped_y.shape[0], self.max_label_size) / float(part_size))), part_size,
                self.max_label_size)
            model = self._model(self.model_store_dir)
            model.fit(csr_mapped_y, mapped_reduced_x, part_size, self.max_label_size)

            mapped_train_predicted_y = model.predict(mapped_reduced_x, predict_cnt)
            mapped_test_predicted_y = model.predict(mapped_reduced_test_x, predict_cnt)

            train_mpr_mre = macro_precision_recall(train_smp.y, y_converter.withdraw_convert(mapped_train_predicted_y),
                                                   min(self.max_label_size, csr_mapped_y.shape[0]))
            test_mpr_mre = macro_precision_recall(test_smp.y, y_converter.withdraw_convert(mapped_test_predicted_y),
                                                  min(self.max_label_size, common_labels_cnt))
            test_f_score = 1. / (1. / test_mpr_mre[0] + 1. / test_mpr_mre[1]) if test_mpr_mre[0] != 0. and test_mpr_mre[
                                                                                                               1] != 0. else float(
                "inf")
            train_f_score = 1. / (1. / train_mpr_mre[0] + 1. / train_mpr_mre[1]) if train_mpr_mre[0] != 0. and \
                                                                                    train_mpr_mre[1] != 0. else float(
                "inf")

            print train_mpr_mre
            print train_f_score

            print test_mpr_mre
            print test_f_score

            if test_f_score > self.best_f_score:
                self.best_f_score = round(test_f_score, 3)
                self.best_x_converter = x_converter
                self.best_y_converter = y_converter
                self.best_threshold = tf_idf_threshold
                self.best_predicted_cnt = predict_cnt
                self.best_model = model

    def __repr__(self):
        return "best_threshold: {0}\nbest_predicted_cnt: {1}".format(self.best_threshold, self.best_predicted_cnt)


if __name__ == '__main__':
    from models.mnb import LaplaceSmoothedMNB

    PATH = sys.argv[1] if len(
        sys.argv) > 1 else '/Users/wumengling/PycharmProjects/kaggle/input_data/small_origin_train_subset.csv'
    cv = PipeLine(LaplaceSmoothedMNB, [90], [1])
    cv.model_selection(PATH, 100, 2)
    print cv.best_predicted_cnt
    print cv.best_f_score
    print cv.best_model
