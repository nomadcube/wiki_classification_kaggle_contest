from classification import Classifier
from Data.Sample import Sample, sample_reader


TF_IDF_THRESHOLD = 0.5
TRAIN_PROP = 0.8
MOST_FREQ_LABEL = '24177'

# read data
dat = sample_reader('/Users/wumengling/PycharmProjects/kaggle/input_data/train_sample.csv')
print(dat.description())

# convert to single label
dat.label_string_disassemble()
print(dat.description())

# remap y
dat.label_string_remap()
print(dat.description())

# convert to binary-class task
# dat.convert_to_binary_class(MOST_FREQ_LABLE)
# print(dat.description())

# perform dimension reduction
dat.dimension_reduction(TF_IDF_THRESHOLD)
print(dat.description())

# split into train and test samples
tr_y, tr_x, te_y, te_x = dat.split_train_test(TRAIN_PROP)

# learning and evaluating
c1 = Classifier.Classifier(tr_y, tr_x, dat.y_remapping_rel)
c1.learn('liblinear')

# for i in range(c1.model.nr_feature):
#     if c1.model.w[i] != 0:
#         print(c1.model.w[i])

print(c1.evaluation(tr_y, tr_x))
print(c1.evaluation(te_y, te_x))
