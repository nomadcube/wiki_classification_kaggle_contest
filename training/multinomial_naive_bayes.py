import StringIO
import cProfile
import pstats

from sklearn.naive_bayes import MultinomialNB

from Data.Sample import sample_reader, construct_csr
from Data.hierarchy import hierarchy
from model_evaluation.evaluation import PredictResult

pr = cProfile.Profile()
pr.enable()

hierarchy_f_path = '/Users/wumengling/PycharmProjects/kaggle/input_data/hierarchy.txt'
sample_f_path = '/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv'
tf_idf_threshold = 1.0
train_prop = 1.0
hierarchy_upward_step = 200
sampling_prop = 0.01

# construct upward hierarchy
hierarchy_table = hierarchy.HierarchyTable()
hierarchy_table.read_data(hierarchy_f_path)
hierarchy_table.update(hierarchy_upward_step)

# read data from file and initiate a new Prediction object
dat = sample_reader(sample_f_path, sampling_prop)
print(dat.description())
predict_result = PredictResult()

# perform dimension reduction
dat.dimension_reduction(tf_idf_threshold)
print(dat.description())
dat.collect_all_feature_and_count()
dat.remap_feature_index_after_tfidf()
print(dat.description())

# disassemble labels
# dat.label_string_disassemble()
# print(dat.description())

# label upward
dat.label_upward(hierarchy_table)
print(dat.description())

# generate train and test data
tr_y, tr_x, te_y, te_x, tr_keys, te_keys = dat.split_train_test(train_prop)

# learn with multinomial naive bayes
nb_tr_x = construct_csr(tr_keys, tr_x)
nb_tr_x_features = set()
max_feature = 0
for x in tr_x:
    for feat in x.keys():
        nb_tr_x_features.add(feat)
        if feat > max_feature:
            max_feature = feat
print("max feature is " + repr(max_feature))
nb_test_x = construct_csr(te_keys, te_x, nb_tr_x_features)
print("nb learning starts")
nb_learner = MultinomialNB()
nb_learner.fit(nb_tr_x, tr_y)
nb_predict_on_train = nb_learner.predict(nb_tr_x)
# nb_predict_on_test = nb_learner.predict(nb_test_x)
print((nb_predict_on_train != tr_y).sum())
print(len(tr_y))
# print((nb_predict_on_test != te_y).sum())
# print(len(te_y))

pr.disable()
s = StringIO.StringIO()
sort_key = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sort_key)
ps.print_stats()
print s.getvalue()
