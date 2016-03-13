import numpy as np

from read import Sample
from transformation.converter import XConverter
from models.mnb import CNB, LaplaceSmoothedMNB
from models.adaboost import AdaBoost
from metrics import evaluation

train_file = '/Users/wumengling/PycharmProjects/kaggle/input_data/sub_train.csv'
tf_idf_threshold = 99

smp = Sample()
smp.read(train_file)
train_smp, cv_smp = smp.extract_and_update()

x_converter = XConverter(train_smp.x, tf_idf_threshold)
train_x = x_converter.convert(train_smp.x)
cv_x = x_converter.convert(cv_smp.x)

model = AdaBoost(LaplaceSmoothedMNB, 11)
model.fit(train_smp.y, train_x)
prediction = model.predict(cv_x)

print np.array(prediction == cv_smp.y, dtype='int8').sum() / float(len(cv_smp.y))
print list(prediction).count(1.0)
