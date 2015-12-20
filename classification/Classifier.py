import liblinearutil
import re
from data_processing.tf_idf import tf_idf
from model_evaluation.evaluation import macro_metric
from feature_extraction.feature_extraction import extraction
from data_analysis.x_description import x_feature_set


class Classifier:
    """Represent one classifier, with certain model and learning algorithm."""

    def __init__(self, train_data, prediction_path, test_data=None, if_dim_reduction=True, if_y_remapped=True):
        self._train_data = train_data
        self._test_data = test_data if test_data else train_data
        self._prediction_path = prediction_path
        self._if_dim_reduction = if_dim_reduction
        self._if_y_remapped = if_y_remapped
        self._dim_reduced_x = None
        self.model = None
        self.evaluation_metric = None

    def learn(self, algorithm='liblinear', threshold=0.0):
        if self._if_dim_reduction:
            x_tf_idf = tf_idf(self._train_data.x)
            self._dim_reduced_x = extraction(x_tf_idf, threshold=threshold)
            x_train = self._dim_reduced_x
        else:
            x_train = self._train_data.x
        y_train = self._train_data.y_remapped() if self._if_y_remapped else self._train_data.y
        if algorithm == 'liblinear':
            self.model = liblinearutil.train(y_train.values(), x_train.values(), '-s 0 -c 1')
        return self.model

    def evaluation(self, on_test=True):
        y_remapped_rel = self._train_data.y_mapping_relation()
        inverse_y_remapped_rel = {v: k for (k, v) in y_remapped_rel.items()}
        dat = self._test_data if on_test else self._train_data
        if self._if_dim_reduction:
            train_x_subset = x_feature_set(self._dim_reduced_x)
            x_test = extraction(tf_idf(dat.x), x_subset=train_x_subset)
        else:
            x_test = dat.x
        y_test = dat.y_remapped() if self._if_y_remapped else dat.y
        self._make_prediction(y_test.values(), x_test.values(), inverse_y_remapped_rel)
        self.evaluation_metric = macro_metric(dat.y, self._prediction_path)

    def _make_prediction(self, y, x, remapped_rel):
        predicted_y, p_acc, p_val = liblinearutil.predict(y, x, self.model)
        with open(self._prediction_path, 'w') as predict_data:
            for index, each_predicted_y in enumerate(predicted_y):
                new_each_predicted_y = re.sub(',', ' ', remapped_rel[each_predicted_y]) if self._if_y_remapped \
                    else each_predicted_y
                predict_data.write(str(index) + ',' + str(new_each_predicted_y) + '\n')
                predict_data.flush()


if __name__ == '__main__':
    from data_processing.TrainData import TrainData
    import os
    train_data_path = '/Users/wumengling/PycharmProjects/kaggle/input_data/train_sample.csv'
    test_data_path = '/Users/wumengling/PycharmProjects/kaggle/input_data/test_sample.csv'
    train_data_line = 100
    test_data_line = 100
    split_train_cmd = "sed -n '1," + str(train_data_line) + \
                      "p' /Users/wumengling/PycharmProjects/kaggle/input_data/train.csv | cat >" + train_data_path

    split_test_cmd = "sed -n '" + str(train_data_line + 1) + "," + str(train_data_line + test_data_line) + \
                     "p' /Users/wumengling/PycharmProjects/kaggle/input_data/train.csv | cat >" + test_data_path
    os.system(split_train_cmd)
    os.system(split_test_cmd)
    c1 = Classifier(TrainData(train_data_path),
                    '/Users/wumengling/PycharmProjects/kaggle/output_data/model_fitting_predict.txt',
                    TrainData(test_data_path))
    c1.learn()
    c1.evaluation()
    print(c1.evaluation_metric)
    c1.evaluation(False)
    print(c1.evaluation_metric)
    for i in range(len(c1.model.w)):
        print(c1.model.w[i])
