# coding=utf-8

import numpy as np
import sys
import math
from memory_profiler import profile
from itertools import product

from read import Sample
from preprocessing.transforming import YConverter, XConverter, convert_y_to_csr
from metrics import get_evaluation_metrics
from preprocessing.tf_idf import tf_idf
from data_analysis.labels import most_frequent_label


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
        self.max_label_size = max_label_size  # 用来限制测试程序时的类别个数，正式预测时设为40w

    # @profile
    def model_selection(self, in_path, part_size, test_path):
        smp = Sample('model_selection')
        smp.read(in_path)
        max_label_in_smp = max([l[0] for l in smp.y])
        train_smp, cv_smp, common_labels_cnt = smp.extract_and_update()
        print most_frequent_label(train_smp.y, 10)
        print most_frequent_label(cv_smp.y, 10)

        y_converter = YConverter()
        y_converter.construct(train_smp.y)
        y_train = y_converter.convert(train_smp.y)

        for param in product(self._threshold, self._predict_cnt):
            tf_idf_threshold, predict_cnt = param

            x_converter = XConverter(tf_idf_threshold)
            x_converter.construct(smp.x)

            x_train = x_converter.convert(train_smp.x)
            x_cv = x_converter.convert(cv_smp.x)

            y_train_csr = convert_y_to_csr(y_train)
            print "num of labels in train set: {0}\n" \
                  "\ntrain set size: {1}\ncv set size:{3}" \
                  "\nfeature count: {2}\n".format(y_train_csr.shape[0], y_train_csr.shape[1],
                                                  x_train.shape[1],
                                                  x_cv.shape[0])
            print "max size of y is {2}\nall y split into {0} parts, each with at most {1} labels\n".format(
                int(math.ceil(min(y_train_csr.shape[0], self.max_label_size) / float(part_size))), part_size,
                self.max_label_size)

            model = self._model(self.model_store_dir)
            model.fit(y_train_csr, x_train, part_size, self.max_label_size)

            prediction_train = y_converter.withdraw_convert(model.predict(x_train, predict_cnt))
            prediction_cv = y_converter.withdraw_convert(model.predict(x_cv, predict_cnt))

            # print most_frequent_label(prediction_cv, 10)

            mat_shape = max_label_in_smp
            metrics_denominator = min(common_labels_cnt, len(np.unique([label[0] for label in train_smp.y])))
            result_train = get_evaluation_metrics(train_smp.y, prediction_train, mat_shape, metrics_denominator)
            result_cv = get_evaluation_metrics(cv_smp.y, prediction_cv, mat_shape, metrics_denominator)

            print result_train
            print result_cv

            if result_cv.f_score > self.best_f_score:
                self.best_f_score = round(result_cv.f_score, 3)
                self.best_x_converter = x_converter
                self.best_y_converter = y_converter
                self.best_threshold = tf_idf_threshold
                self.best_predicted_cnt = predict_cnt
                self.best_model = model

        result_test = self._evaluation(test_path, max_label_in_smp)
        print result_test

    def _evaluation(self, test_file_path, max_label_in_smp):
        exam_smp = Sample('submission')
        exam_smp.read(test_file_path)
        transformed_x = self.best_x_converter.convert(exam_smp.x)
        predicted_y = self.best_model.predict(transformed_x, self.best_predicted_cnt)
        origin_predicted_y = self.best_y_converter.withdraw_convert(predicted_y)
        return get_evaluation_metrics(exam_smp.y, origin_predicted_y, max_label_in_smp)

    @staticmethod
    def submission(origin_predicted_y, output_file_path):
        with open(output_file_path, 'w') as out:
            out.write('Id,Predicted' + '\n')
            for i, each_predicted_y in enumerate(origin_predicted_y):
                out.write("{0},{1}\n".format(i, ' '.join([str(i) for i in each_predicted_y])))
            out.flush()

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
