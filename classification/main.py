from Data.Sample import sample_reader
from model_evaluation.evaluation import Prediction, generate_fact
import liblinearutil


def update_by_one_label(current_predict_result, whole_sample, target_label, train_proportion):
    whole_sample.convert_to_binary_class(target_label)
    tr_y, tr_x, te_y, te_x, tr_keys, te_keys = whole_sample.split_train_test(train_proportion)
    model = liblinearutil.train(tr_y, tr_x, '-s 0 -c 1')
    predicted_y, accuracy, decision_value = liblinearutil.predict(tr_y, tr_x, model)
    current_predict_result.update(target_label, tr_keys, predicted_y)


def select_labels_for_prediction(predicting_label_count):
    distinct_labels = list()
    for each_y in dat.y.values():
        if len(distinct_labels) >= predicting_label_count:
            break
        if each_y not in distinct_labels:
            distinct_labels.append(each_y)
    return distinct_labels


TF_IDF_THRESHOLD = 0.5
TRAIN_PROP = 0.8
PREDICTING_LABEL_NUM = 30
predict_result = Prediction()

dat = sample_reader('/Users/wumengling/PycharmProjects/kaggle/input_data/train_sample.csv')
dat.label_string_disassemble()
dat.dimension_reduction(TF_IDF_THRESHOLD)

print(dat.description().class_number)

prediction_labels = select_labels_for_prediction(dat.description().class_number)
for each_label in prediction_labels:
    update_by_one_label(predict_result, dat, each_label, 1.0)

predict_result.convert_to_original_index(dat.index_mapping_relation)
dat_fact = generate_fact(dat.y, dat.index_mapping_relation)
print(dat_fact)
print(predict_result.dat)
print(predict_result.evaluation(dat_fact))
