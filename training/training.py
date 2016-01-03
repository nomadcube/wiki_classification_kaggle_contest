import liblinearutil


def select_labels_for_prediction(sample, predicting_label_count):
    distinct_labels = list()
    for each_y in sample.y.values():
        if len(distinct_labels) >= predicting_label_count:
            break
        if each_y not in distinct_labels:
            distinct_labels.append(each_y)
    return distinct_labels


def train_and_collect_predict_result(current_predict_result, whole_sample, target_label, train_proportion):
    whole_sample.convert_to_binary_class(target_label)
    tr_y, tr_x, te_y, te_x, tr_keys, te_keys = whole_sample.split_train_test(train_proportion)
    model = liblinearutil.train(tr_y, tr_x, '-s 0 -c 0.03')
    predicted_y, accuracy, decision_value = liblinearutil.predict(te_y, te_x, model)
    current_predict_result.update(target_label, te_keys, predicted_y)
