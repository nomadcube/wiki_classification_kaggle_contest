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
    def __init__(self, model_class, threshold, predict_cnt, model_store_dir, test_data_store_dir):
        self._model = model_class
        self._threshold = threshold
        self._predict_cnt = predict_cnt

        self.best_f_score = 0.
        self.best_x_converter = None
        self.best_y_converter = None
        self.best_threshold = None
        self.best_predicted_cnt = None
        self.best_model = None

        self.model_store_dir = model_store_dir
        self.test_data_store_dir = test_data_store_dir

    # @profile
    def model_selection(self, in_path, part_size):
        smp = Sample()
        smp.read(in_path)
        train_smp, test_smp, common_labels_cnt = smp.extract_and_update()
        # 35053209 function calls in 278.926 seconds (on train.csv)

        # y_converter = YConverter()
        # y_converter.construct(train_smp.y)
        # mapped_y = y_converter.convert(train_smp.y)
        #
        # for param in product(self._threshold, self._predict_cnt):
        #     tf_idf_threshold, predict_cnt = param
        #
        #     x_converter = XConverter(tf_idf_threshold)
        #     x_converter.construct(train_smp.x)
        #
        #     mapped_reduced_x = tf_idf(x_converter.convert(train_smp.x))
        #     mapped_reduced_test_x = tf_idf(x_converter.convert(test_smp.x))
        #
        #     csr_mapped_y = convert_y_to_csr(mapped_y)
        #     print "num of labels in train set: {0}\nfeature count in train set: {2}\ntrain set size: {1}\ntest set size:{3}".format(
        #         csr_mapped_y.shape[0], csr_mapped_y.shape[1], mapped_reduced_x.shape[1], mapped_reduced_test_x.shape[0])
        #
        #     print "\nall y split into {0} parts, each with at most {1} label".format(
        #         int(math.ceil(csr_mapped_y.shape[0] / float(part_size))), part_size)
        #     model = self._model(self.model_store_dir)
        #     model.fit(csr_mapped_y, mapped_reduced_x, part_size)
        #     mapped_test_predicted_y = model.predict(mapped_reduced_test_x, predict_cnt)
        #
        #     mpr_mre = macro_precision_recall(test_smp.y, y_converter.withdraw_convert(mapped_test_predicted_y),
        #                                      len(y_converter.label_old_new_relation), common_labels_cnt)
        #     f_score = 1. / (1. / mpr_mre[0] + 1. / mpr_mre[1]) if mpr_mre[0] != 0. and mpr_mre[1] != 0. else float(
        #         "inf")
        #     print mpr_mre
        #     print f_score
        #
        #     if f_score > self.best_f_score:
        #         self.best_f_score = round(f_score, 3)
        #         self.best_x_converter = x_converter
        #         self.best_y_converter = y_converter
        #         self.best_threshold = tf_idf_threshold
        #         self.best_predicted_cnt = predict_cnt
        #         self.best_model = model

    def submission(self, test_file_path, output_file_path, transformed_x_exited=False):
        if not transformed_x_exited:
            exam_smp = Sample()
            exam_smp.read(test_file_path)
            transformed_x = self.best_x_converter.convert(exam_smp.x)
            with open('{0}/transformed_x.dat'.format(self.test_data_store_dir), 'wb') as transformed_x_f:
                pickle.dump(transformed_x, transformed_x_f)
            predicted_y = self.best_model.predict(transformed_x, self.best_predicted_cnt)
            origin_predicted_y = self.best_y_converter.withdraw_convert(predicted_y)

            with open(output_file_path, 'w') as out:
                out.write('Id,Predicted' + '\n')
                for i, each_predicted_y in enumerate(origin_predicted_y):
                    out.write("{0},{1}\n".format(i, ' '.join([str(i) for i in each_predicted_y])))
                out.flush()
        else:
            with open('{0}/transformed_x.dat'.format(self.test_data_store_dir), 'r') as transformed_x_f:
                transformed_x = pickle.load(transformed_x_f)
            predicted_y = self.best_model.predict(transformed_x, self.best_predicted_cnt)
            origin_predicted_y = self.best_y_converter.withdraw_convert(predicted_y)

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
