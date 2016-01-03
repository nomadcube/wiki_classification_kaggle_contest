from Data.Sample import sample_reader
from model_evaluation.evaluation import PredictResult, generate_real_class
from Data.hierarchy import hierarchy
from training import select_labels_for_prediction, train_and_collect_predict_result

import cProfile
import pstats
import StringIO

pr = cProfile.Profile()
pr.enable()


# ---------------------------------- main part -------------------------- #
hierarchy_f_path = '/Users/wumengling/PycharmProjects/kaggle/input_data/hierarchy.txt'
sample_f_path = '/Users/wumengling/PycharmProjects/kaggle/input_data/train_sample.csv'
tf_idf_threshold = 1.0
train_prop = 0.8
num_labels_to_be_predicted = 2
hierarchy_upward_step = 100


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

# disassemble label string
dat.label_string_disassemble()
print(dat.description())

# label upward
dat.disassembled_label_upward(hierarchy_table)
print(dat.description())

# training
labels_to_be_predicted = select_labels_for_prediction(dat, num_labels_to_be_predicted)
for each_label in labels_to_be_predicted:
    train_and_collect_predict_result(predict_result, dat, each_label, train_prop)

# evaluating with macro metrics
predict_result.convert_to_original_index(dat.index_mapping_relation)
dat_fact = generate_real_class(dat.y, dat.index_mapping_relation)
print(predict_result.evaluation(dat_fact, labels_to_be_predicted))
# ---------------------------------- main part -------------------------- #


pr.disable()
s = StringIO.StringIO()
sort_key = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sort_key)
ps.print_stats()
# print s.getvalue()
