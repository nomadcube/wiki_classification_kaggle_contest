import StringIO
import cProfile
import pstats
import sys
from learning.chunking import iter_chuck
import numpy as np
import memory_profiler

from data_processing.transformation import base_sample_reader, construct_csr_sample, x_with_tf_idf
from model_evaluation.cross_validation import split_train_test
from model_evaluation.evaluation_metrics import macro_precision_and_recall
from learning.redefined_multinomial_naive_bayes import MultiOutputMultinomialNB
from data_processing.description import describe_x_y
from sklearn.preprocessing import MultiLabelBinarizer

submission_test_path = '/Users/wumengling/PycharmProjects/kaggle/input_data/test.csv'
predict_result_path = '/Users/wumengling/PycharmProjects/kaggle/input_data/predict_result.csv'
log_f_path = '/Users/wumengling/PycharmProjects/kaggle/log/learning_log.txt'

learning_config = {'sample_proportion': 1,
                   'tf_idf_threshold': 0.1,
                   'split_proportion': 0.8,
                   'predict_class_num': 3}

sample = base_sample_reader('/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv',
                            learning_config['sample_proportion'])

# dimension_reduced_x = x_with_tf_idf(sample.x, learning_config['tf_idf_threshold']) if \
#     learning_config['tf_idf_threshold'] > 0 else sample.x

# train_y, train_x, test_y, test_x = split_train_test(dimension_reduced_x, sample.y, learning_config['split_proportion'])
# train_desc = describe_x_y(train_x, train_y)
# learning_config['train_sample_size'] = train_desc.sample_size
# learning_config['train_feature_dimension'] = train_desc.feature_dimension

# csr_train_x, csr_feature_mapping_rel = construct_csr_sample(train_x)
#
# all_classes = train_desc.class_distribution.keys()
# nb_learner = MultiOutputMultinomialNB()
# for chuck_train_y, chuck_train_x in chunking.iter_chuck(train_y.__iter__(), train_x.__iter__(), 200000):
#     flat_csr_train_x = construct_csr_sample(chuck_train_x, csr_feature_mapping_rel)
#     nb_learner.partial_fit(flat_csr_train_x, chuck_train_y, classes=all_classes)
#
# predicted_y_train = nb_learner.multi_predict(csr_train_x, 3)
# print(macro_precision_and_recall([','.join(each_y) for each_y in train_y], predicted_y_train))
#
# predicted_y_test = nb_learner.multi_predict(construct_csr_sample(test_x, csr_feature_mapping_rel), 3)
# print(macro_precision_and_recall([','.join(each_y) for each_y in test_y], predicted_y_test))
