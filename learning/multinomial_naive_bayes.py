import StringIO
import cProfile
import pstats

from sklearn.naive_bayes import MultinomialNB

from data.transformation import base_sample_reader, construct_csr_sample, x_with_tf_idf, flatting_multi_label, \
    assemble_y
from model_evaluation.cross_validation import split_train_test
from model_evaluation.evaluation_metrics import macro_precision_and_recall

pr = cProfile.Profile()
pr.enable()

# ---------------------------------- main part -------------------------- #
sample = base_sample_reader('/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv', 0.0005)
dimension_reduced_x = x_with_tf_idf(sample.x, 0)
train_y, train_x, test_y, test_x = split_train_test(dimension_reduced_x, sample.y, 0.8)
flat_train_y, flat_train_x, flat_train_rel = flatting_multi_label(train_y, train_x)

flat_csr_train_x, feat_mapping_rel = construct_csr_sample(flat_train_x)

nb_learner = MultinomialNB()
nb_learner.fit(flat_csr_train_x, flat_train_y)

# evaluation on train data
nb_predict_on_train = nb_learner.predict(flat_csr_train_x)
train_prediction_cmp = (flat_train_y == nb_predict_on_train)
print("accuracy: {0}, {1} out of {2}.".format(float(sum(train_prediction_cmp)) / float(len(flat_train_y)),
                                              sum(train_prediction_cmp), len(flat_train_y)))
print(
macro_precision_and_recall(assemble_y(flat_train_y, flat_train_rel), assemble_y(nb_predict_on_train, flat_train_rel)))

# evaluation on test data
flat_test_y, flat_test_x, flat_test_rel = flatting_multi_label(test_y, test_x)
flat_csr_test_x = construct_csr_sample(flat_test_x, feat_mapping_rel)
nb_predict_on_test = nb_learner.predict(flat_csr_test_x)
test_prediction_cmp = (flat_test_y == nb_predict_on_test)
print("accuracy: {0}, {1} out of {2}.".format(float(sum(test_prediction_cmp)) / float(len(flat_test_y)),
                                              sum(test_prediction_cmp), len(flat_test_y)))
print(macro_precision_and_recall(assemble_y(flat_test_y, flat_test_rel), assemble_y(nb_predict_on_test, flat_test_rel)))
# ---------------------------------- main part -------------------------- #


pr.disable()
s = StringIO.StringIO()
sort_key = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sort_key)
ps.print_stats()
print s.getvalue()
