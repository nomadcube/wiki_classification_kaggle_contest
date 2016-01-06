import StringIO
import cProfile
import pstats

from sklearn.naive_bayes import MultinomialNB

from data.transformation import base_sample_reader, construct_csr_sample, x_with_tf_idf
from model_evaluation.cross_validation import split_train_test
from model_evaluation.evaluation_metrics import macro_precision_and_recall

pr = cProfile.Profile()
pr.enable()

# ---------------------------------- main part -------------------------- #
sample = base_sample_reader('/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv', 0.002)
dimension_reduced_x = x_with_tf_idf(sample.x, 0.2)
train_y, train_x, test_y, test_x = split_train_test(dimension_reduced_x, sample.y, 0.8)

csr_train_x, feat_mapping_rel = construct_csr_sample(train_x)
csr_test_x = construct_csr_sample(test_x, feat_mapping_rel)

nb_learner = MultinomialNB()
nb_learner.fit(csr_train_x, train_y)

nb_predict_on_train = nb_learner.predict(csr_train_x)
train_prediction_cmp = (train_y == nb_predict_on_train)
print("accuracy: {0}, {1} out of {2}.".format(float(sum(train_prediction_cmp)) / float(len(train_y)),
                                              sum(train_prediction_cmp), len(train_y)))
print(macro_precision_and_recall(train_y, nb_predict_on_train))

nb_predict_on_test = nb_learner.predict(csr_test_x)
train_prediction_cmp = (test_y == nb_predict_on_test)
print("accuracy: {0}, {1} out of {2}.".format(float(sum(train_prediction_cmp)) / float(len(test_y)),
                                              sum(train_prediction_cmp), len(test_y)))
print(macro_precision_and_recall(test_y, nb_predict_on_train))
# ---------------------------------- main part -------------------------- #


pr.disable()
s = StringIO.StringIO()
sort_key = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sort_key)
ps.print_stats()
# print s.getvalue()
