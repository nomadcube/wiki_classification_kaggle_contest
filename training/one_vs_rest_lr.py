import StringIO
import cProfile
import pstats

import liblinearutil

from Data.Sample import sample_reader
from Data.hierarchy import hierarchy
from model_evaluation.evaluation import PredictResult, generate_real_class
from training import select_labels_for_prediction

pr = cProfile.Profile()
pr.enable()


# ---------------------------------- main part -------------------------- #
hierarchy_f_path = '/Users/wumengling/PycharmProjects/kaggle/input_data/hierarchy.txt'
sample_f_path = '/Users/wumengling/PycharmProjects/kaggle/input_data/train_sample.csv'
tf_idf_threshold = 1.0
train_prop = 0.8
num_labels_to_be_predicted = 1
hierarchy_upward_step = 0


# construct upward hierarchy
hierarchy_table = hierarchy.HierarchyTable()
hierarchy_table.read_data(hierarchy_f_path)
hierarchy_table.update(hierarchy_upward_step)

# read data from file and initiate a new Prediction object
dat = sample_reader(sample_f_path)
predict_result = PredictResult()

# perform dimension reduction
dat.dimension_reduction(tf_idf_threshold)
print(dat.description())

# label upward
# dat.label_upward(hierarchy_table)
# print(dat.description())

# training
labels_to_be_predicted = select_labels_for_prediction(dat, num_labels_to_be_predicted)
print("label " + repr(labels_to_be_predicted) + " is training.")
for each_label in labels_to_be_predicted:
    dat.convert_to_binary_class(each_label)
    tr_y, tr_x, te_y, te_x, tr_keys, te_keys = dat.split_train_test(train_prop)
    model = liblinearutil.train(tr_y, tr_x, '-s 0 -c 0.03')
    predicted_y = liblinearutil.predict(te_y, te_x, model)[0]
    predict_result.update(each_label, te_keys, predicted_y)
    test_y_sample = {k: dat.y[k] for k in te_keys}
    dat_fact = generate_real_class(test_y_sample)
    print(predict_result.evaluation(dat_fact, labels_to_be_predicted))
# ---------------------------------- main part -------------------------- #


pr.disable()
s = StringIO.StringIO()
sort_key = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sort_key)
ps.print_stats()
# print s.getvalue()
