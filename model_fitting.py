from data_processing.TrainData import TrainData
from model_evaluation.evaluation import macro_metric
from data_processing.tf_idf import dim_reduction_with_tf_idf


import liblinearutil
import re
import sys


train_data_path = sys.argv[1]
predict_data_path = '/Users/wumengling/PycharmProjects/kaggle/predict_output_data/model_fitting_predict.txt'


def training(y_train, x_train):
    return liblinearutil.train(y_train, x_train, '-s 0 -c 1 -q')


def predicting(y_test, x_test, model, predict_path, y_remapped_rel):
    """Make prediction on [y_test, x_test] with model generated in training."""
    inverse_y_remapped_rel = {v: k for (k, v) in y_remapped_rel.items()}
    p_label, p_acc, p_val = liblinearutil.predict(y_test, x_test, model)
    with open(predict_path, 'w') as predict_data:
        for index, predicted_label in enumerate(p_label):
            predict_data.write(str(index) + ',' + re.sub(',', '', inverse_y_remapped_rel[predicted_label]) + '\n')
            predict_data.flush()


def learning_utils(train_path, predict_path):
    train_dat = TrainData(train_path)
    dim_reduction_x = dim_reduction_with_tf_idf(train_dat.x, train_dat.feature_set, 0)
    y = [y_val for y_val in train_dat.y_remapped().values()]
    print(y)
    x = [x_val for x_val in dim_reduction_x.values()]
    print(x)
    m = training(y, x)
    predicting(y, x, m, predict_path, train_dat.y_mapping_relation)
    return macro_metric(train_path, predict_path)


print(learning_utils(train_data_path, predict_data_path))

