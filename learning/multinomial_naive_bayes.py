import StringIO
import cProfile
import pstats

from sklearn.naive_bayes import MultinomialNB

from data.transformation import base_sample_reader, construct_csr_sample, x_with_tf_idf, flatting_multi_label, \
    assemble_y
from model_evaluation.cross_validation import split_train_test
from model_evaluation.evaluation_metrics import macro_precision_and_recall
from redefined_multinomial_naive_bayes import MultiClassMultinomialNB
from data.description import describe_x_y

pr = cProfile.Profile()
pr.enable()

# ---------------------------------- main part -------------------------- #
sample = base_sample_reader('/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv', 0.001)
dimension_reduced_x = x_with_tf_idf(sample.x, 0)
train_y, train_x, test_y, test_x = split_train_test(dimension_reduced_x, sample.y, 0.8)
flat_train_y, flat_train_x, flat_train_rel = flatting_multi_label(train_y, train_x)
desc_res = describe_x_y(flat_train_x, flat_train_y)

flat_csr_train_x, feat_mapping_rel = construct_csr_sample(flat_train_x)

nb_learner = MultiClassMultinomialNB()
nb_learner.fit(flat_csr_train_x, flat_train_y)

# evaluation on train data
nb_predict_on_train = nb_learner.multi_predict(flat_csr_train_x, 3)
print(macro_precision_and_recall(train_y, assemble_y(nb_predict_on_train, flat_train_rel)))

for most_predict_label in assemble_y(nb_predict_on_train, flat_train_rel)[0].split(','):
    print(desc_res.class_distribution[most_predict_label])
print(sorted(desc_res.class_distribution.values(), reverse=True)[:10])

# evaluation on test data
flat_test_y, flat_test_x, flat_test_rel = flatting_multi_label(test_y, test_x)
flat_csr_test_x = construct_csr_sample(flat_test_x, feat_mapping_rel)
nb_predict_on_test = nb_learner.multi_predict(flat_csr_test_x, 3)
print(macro_precision_and_recall(test_y, assemble_y(nb_predict_on_test, flat_test_rel)))
# ---------------------------------- main part -------------------------- #


pr.disable()
s = StringIO.StringIO()
sort_key = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sort_key)
ps.print_stats()
print s.getvalue()
