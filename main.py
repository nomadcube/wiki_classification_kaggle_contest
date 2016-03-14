import numpy as np

from read import Sample
from transformation.converter import XConverter
from models.mnb import CNB, LaplaceSmoothedMNB, NonSmoothedMNB
from models.adaboost import AdaBoost
from metrics import evaluation

train_file = '/Users/wumengling/PycharmProjects/kaggle/input_data/sub_train.csv'
test_file = '/Users/wumengling/PycharmProjects/kaggle/input_data/sub_test.csv'
tf_idf_threshold = 99

smp = Sample()
smp.read(train_file)
test_smp = Sample()
test_smp.read(test_file)

train_smp, cv_smp = smp.extract_and_update()

x_converter = XConverter(train_smp.x, tf_idf_threshold)
train_x = x_converter.convert(train_smp.x)
cv_x = x_converter.convert(cv_smp.x)
test_x = x_converter.convert(test_smp.x)

model = AdaBoost(NonSmoothedMNB, 10)
model.fit(train_smp.y, train_x)
prediction = model.predict(cv_x)
print evaluation(cv_smp.y, prediction)

test_prediction = model.predict(test_x)
print evaluation(test_smp.y, test_prediction)
