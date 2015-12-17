from data_processing.tf_idf import term_frequency, log_inverse_doc_frequency
from data_processing.libsvm_train_data import training_label, dimension_reduction_instance
from model_fitting import training, predicting
from model_evaluation.evaluation import macro_metric, confusion_matrix_per_label, true_id_per_label

# todo: 12.16 19:31 just did some dirty thing to make whole program run. need to be fix.
train_data_path = '/Users/wumengling/kaggle/unit_test_data/train_1000.csv'
predict_data_path = '/Users/wumengling/kaggle/unit_test_data/fitting_predict.txt'

# y = [i for i in training_label(train_data_path)]
# print(len(y))
# print(y[:10])
# tf = term_frequency(train_data_path)
# print(len(tf))
# idf = log_inverse_doc_frequency(train_data_path)
# print(len(idf))
# x = [i for i in dimension_reduction_instance(tf, idf, 0)]
# print(len(x))
# lr_model = training(y, x)
# predicting(y, x, lr_model, predict_data_path)
#
#
each_measure = confusion_matrix_per_label(predict_data_path, true_id_per_label(train_data_path))
print(macro_metric(each_measure))   # todo: macro precision and recall are all too low.
