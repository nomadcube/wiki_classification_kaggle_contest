from collections import namedtuple


def index_for_each_label(y):
    res = dict()
    for instance_index, labels in enumerate(y):
        for each_label in labels.split(','):
            res.setdefault(each_label, set())
            res[each_label].add(instance_index)
    return res


def all_index(real_class):
    res = set()
    for index in real_class.values():
        for each_index in index:
            res.add(each_index)
    return res


def confusion_matrix(real_index_for_one_label, predicted_index_for_one_label, whole_index):
    measure = namedtuple('measure', 'true_pos false_pos true_neg false_neg')
    true_pos, false_pos, true_neg, false_neg = [0.0] * 4
    for each_index in whole_index:
        if each_index in predicted_index_for_one_label:
            if each_index in real_index_for_one_label:
                true_pos += 1
            else:
                false_pos += 1
        else:
            if each_index in real_index_for_one_label:
                false_neg += 1
            else:
                true_neg += 1
    return measure(true_pos, false_pos, true_neg, false_neg)


def precision_and_recall(confusion_mat):
    pos_num = confusion_mat.true_pos + confusion_mat.false_pos
    true_num = confusion_mat.true_pos + confusion_mat.false_neg
    if pos_num != 0 and true_num != 0:
        precision = confusion_mat.true_pos / pos_num
        recall = confusion_mat.true_pos / true_num
        return precision, recall
    else:
        return 0.0, 0.0


def macro_precision_and_recall(y, prediction):
    all_precision = list()
    all_recall = list()
    real_label_index = index_for_each_label(y)
    predict_label_index = index_for_each_label(prediction)
    whole_index = all_index(real_label_index)
    for each_label in real_label_index.keys():
        if predict_label_index.get(each_label) is None:
            continue
        confusion_mat = confusion_matrix(real_label_index[each_label], predict_label_index[each_label], whole_index)
        tmp_precision, tmp_recall = precision_and_recall(confusion_mat)
        all_precision.append(tmp_precision)
        all_recall.append(tmp_recall)
    return float(sum(all_precision)) / len(all_precision), float(
        sum(all_recall) / len(all_recall)), all_precision, all_recall
