import sys
import numpy as np

from read_data import read_sparse
from transformation.converter import XConverter
from models.mnb import CNB, LaplaceSmoothedMNB, WeightedNonSmoothedMNB
from models.adaboost import AdaBoost
from evaluation import evaluation
from data_analysis.labels import occurrence

train_file = sys.argv[1] if len(sys.argv) > 1 else '/Users/wumengling/PycharmProjects/kaggle/input_data/sub_train.csv'
test_file = sys.argv[2] if len(sys.argv) > 2 else '/Users/wumengling/PycharmProjects/kaggle/input_data/sub_test.csv'
tf_idf_threshold = 99.2

smp = read_sparse(train_file)
smp.convert_to_tri_class('24177', '203001')
test_smp = read_sparse(test_file)
test_smp.convert_to_tri_class('24177', '203001')

train_smp, cv_smp = smp.train_cv_split()

x_converter = XConverter(train_smp.x, tf_idf_threshold)
train_x = x_converter.convert(train_smp.x)
cv_x = x_converter.convert(cv_smp.x)
test_x = x_converter.convert(test_smp.x)

model = LaplaceSmoothedMNB()
model.fit(train_smp.y, train_x)
prediction = model.predict(cv_x)
print evaluation(cv_smp.y, prediction)
#
# test_prediction = model.predict(test_x)
# print evaluation(test_smp.y, test_prediction)
