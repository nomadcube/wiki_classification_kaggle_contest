import StringIO
import cProfile
import pstats

from sklearn.naive_bayes import MultinomialNB

from data.transformation import base_sample_reader, construct_csr_sample, x_with_tf_idf
from model_evaluation.evaluation_metrics import macro_precision_and_recall

pr = cProfile.Profile()
pr.enable()

# ---------------------------------- main part -------------------------- #
sample = base_sample_reader('/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv', 0.004)
dimension_reduced_x = x_with_tf_idf(sample.x, 0.2)
train_x = construct_csr_sample(dimension_reduced_x)
train_y = sample.y

nb_learner = MultinomialNB()
nb_learner.fit(train_x, train_y)
nb_predict_on_train = nb_learner.predict(train_x)
train_prediction_cmp = (train_y == nb_predict_on_train)
print("accuracy: {0}, {1} out of {2}.".format(float(sum(train_prediction_cmp)) / float(len(train_y)),
                                              sum(train_prediction_cmp), len(train_y)))
print(macro_precision_and_recall(train_y, nb_predict_on_train))
# ---------------------------------- main part -------------------------- #


pr.disable()
s = StringIO.StringIO()
sort_key = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sort_key)
ps.print_stats()
# print s.getvalue()
