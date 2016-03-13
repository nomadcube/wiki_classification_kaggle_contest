from read import Sample
from transformation.converter import XConverter
from models.mnb import CNB, LaplaceSmoothedMNB
from metrics import evaluation

train_file = '/Users/wumengling/PycharmProjects/kaggle/input_data/sub_train.csv'
tf_idf_threshold = 90

smp = Sample()
smp.read(train_file)
train_smp, cv_smp = smp.extract_and_update()

x_converter = XConverter(train_smp.x, tf_idf_threshold)
train_x = x_converter.convert(train_smp.x)
cv_x = x_converter.convert(cv_smp.x)

model = LaplaceSmoothedMNB()
model.fit(train_smp.y, train_x)
prediction = model.predict(train_x)

print evaluation(train_smp.y, prediction)
