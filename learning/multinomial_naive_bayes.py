import sys

from time import time
from data_processing.transformation import sample_reader

path_train_sample = sys.argv[1] if len(
        sys.argv) >= 2 else '/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv'

max_line = 2365436
sample_size = 1000000

start_time = time()
samples = sample_reader(path_train_sample, max_line, sample_size)
print(time() - start_time)
print(type(samples.y))
print(len(samples.y))


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
