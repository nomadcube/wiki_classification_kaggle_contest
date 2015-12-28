from classification import Classifier
from Data.Data import Data


TF_IDF_THRESHOLD = 0.5
TRAIN_COUNT = 0.8
LABEL_MODE = '24177'

# read data
dat = Data('/Users/wumengling/PycharmProjects/kaggle/input_data/train_sample.csv')
print(dat.description())

# convert to binary-class task
dat.convert_to_binary_class(LABEL_MODE)

# perform dimension reduction
dat.dim_reduction(TF_IDF_THRESHOLD)
print(dat.description())

# split into train and test samples
dat.sample_split(TRAIN_COUNT)

# learning
c1 = Classifier.Classifier(dat.train_y, dat.train_x, dat.y_remapping_rel)
c1.learn()
print(c1.evaluation(dat.train_y, dat.train_x))
print(c1.evaluation(dat.test_y, dat.test_x))
