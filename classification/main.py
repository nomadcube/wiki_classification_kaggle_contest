from Classifier import Classifier
from Data.TrainData import TrainData
from Data.generate_train_sample import generate_train_sample


# generate train_sample and test_sample
generate_train_sample(100, 100)

# read train data from path
TR = TrainData('/Users/wumengling/PycharmProjects/kaggle/input_data/train_sample.csv')
# remapped train_y
train_y = TR.y.remap().remapped_data
# dimension reduction train_x
train_x = TR.x.dim_reduction(1.0).dim_reduction_data
# y mapping relationship
y_remapping_rel = TR.y.remapping_relation

# test data
TE = TrainData('/Users/wumengling/PycharmProjects/kaggle/input_data/test_sample.csv')
test_y = TE.y.remap(y_remapping_rel).remapped_data
test_x = TE.x.dim_reduction(1.0).dim_reduction_data

# learning
# c1 = Classifier(TR, '/Users/wumengling/PycharmProjects/kaggle/output_data/model_fitting_predict.txt', 1.0)
# print(TR.description())
# c1.learn()
# c1.predict()
# c1.evaluation()
# print(c1.evaluation_metric)

