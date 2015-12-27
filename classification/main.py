from classification import Classifier
from Data.Data import Data
import sys


tf_idf_threshold = 1.0
train_count = 0.8

# read data, remapping and split into train and test samples
dat = Data('/Users/wumengling/PycharmProjects/kaggle/input_data/train_sample.csv')
print(dat.description())
dat.remap()
dat.dim_reduction(tf_idf_threshold)
dat.sample_split(train_count)
print(dat.description())
print(sys.getsizeof(dat))
print(sys.getsizeof(dat.train_y))
print(sys.getsizeof(dat.train_x))

# learning
c1 = Classifier.Classifier(dat.train_y, dat.train_x, dat.y_remapping_rel)
c1.learn()
print(c1.evaluation(dat.test_y, dat.test_x))
