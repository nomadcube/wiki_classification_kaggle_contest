import StringIO
import cProfile
import pstats

import liblinearutil

from data_processing.data_preparation import preparation_for_train
from model_evaluation.evaluation import PredictResult, generate_real_class

pr = cProfile.Profile()
pr.enable()

# ---------------------------------- main part -------------------------- #
config = {
    'hierarchy_f_path': '/Users/wumengling/PycharmProjects/kaggle/input_data/hierarchy.txt',
    'hierarchy_upward_step': 200,
    'sample_f_path': '/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv',
    'sampling_prop': 0.01,
    'tf_idf_threshold': 1.0,
    'train_prop': 1.0,
    'disassemble_multi_label': True,
    'label_upward': True
}

all_data = preparation_for_train(config)
dat = all_data[0]
predict_result = PredictResult()

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
print s.getvalue()
