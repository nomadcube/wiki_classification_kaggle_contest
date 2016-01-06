import StringIO
import cProfile
import pstats

from sklearn.naive_bayes import MultinomialNB

from model_evaluation.evaluation import PredictResult

pr = cProfile.Profile()
pr.enable()

# ---------------------------------- main part -------------------------- #
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
predict_result = PredictResult()
nb_learner = MultinomialNB()
nb_learner.fit(nb_tr_x, tr_y)
nb_predict_on_train = nb_learner.predict(nb_tr_x)
# ---------------------------------- main part -------------------------- #


pr.disable()
s = StringIO.StringIO()
sort_key = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sort_key)
ps.print_stats()
print s.getvalue()
