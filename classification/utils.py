from data_processing.TrainData import TrainData
from Classifier import Classifier
from data_analysis.x_description import x_feature_set
from data_processing.tf_idf import tf_idf
from feature_extraction.feature_extraction import extraction

import os


train_data_path = '/Users/wumengling/PycharmProjects/kaggle/input_data/train_sample.csv'
test_data_path = '/Users/wumengling/PycharmProjects/kaggle/input_data/test_sample.csv'
prediction_path = '/Users/wumengling/PycharmProjects/kaggle/output_data/model_fitting_predict.txt'
train_data_line = 10000
test_data_line = 10000
tf_idf_threshold = 0.5


split_train_cmd = "sed -n '1," + str(train_data_line) + \
                  "p' /Users/wumengling/PycharmProjects/kaggle/input_data/train.csv | cat >" + train_data_path
split_test_cmd = "sed -n '" + str(train_data_line + 1) + "," + str(train_data_line + test_data_line) + \
                 "p' /Users/wumengling/PycharmProjects/kaggle/input_data/train.csv | cat >" + test_data_path

os.system(split_train_cmd)
os.system(split_test_cmd)

train_data = TrainData(train_data_path)
test_data = TrainData(test_data_path)

print(len(x_feature_set(train_data.x)))
print(len(x_feature_set(tf_idf(train_data.x))))
print(len(extraction(tf_idf(train_data.x), threshold=tf_idf_threshold)))


c1 = Classifier(train_data, prediction_path, test_data)
c1.learn(threshold=tf_idf_threshold)
c1.evaluation(on_test=False)
print(c1.evaluation_metric)
