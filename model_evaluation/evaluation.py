from collections import namedtuple


def generate_real_class(y):
    res = dict()
    for instance_index, labels in y.items():
        for each_label in labels.split(','):
            res.setdefault(each_label, set())
            res[each_label].add(instance_index)
    return res


def all_index_in_real_class(real_class):
    res = set()
    for index in real_class.values():
        for each_index in index:
            res.add(each_index)
    return res


def confusion_matrix(true_index, predict_index, whole_index):
    measure = namedtuple('measure', 'true_pos false_pos true_neg false_neg')
    true_pos, false_pos, true_neg, false_neg = [0.0] * 4
    for each_index in whole_index:
        if each_index in predict_index:
            if each_index in true_index:
                true_pos += 1
            else:
                false_pos += 1
        else:
            if each_index in true_index:
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


def macro_precision_and_recall(fact, prediction, predicted_labels):
    macro_precision = 0.0
    macro_recall = 0.0
    whole_index = all_index_in_real_class(fact)
    valid_key = 0.0
    try:
        for each_label in predicted_labels:
            confusion_mat = confusion_matrix(fact[each_label], prediction[each_label], whole_index)
            print(confusion_mat)
            tmp_precision, tmp_recall = precision_and_recall(confusion_mat)
            macro_precision += tmp_precision
            macro_recall += tmp_recall
            valid_key += 1
        return macro_precision / valid_key, macro_recall / valid_key
    except KeyError:
        return "Macro metrics could be infinity because some label are not included in prediction."


class PredictResult:
    def __init__(self):
        self.dat = dict()

    def update(self, target_label, predicting_sample_keys, predicted_label):
        if len(predicting_sample_keys) != len(predicted_label):
            raise ValueError('The length of predicting sample keys and predicted label must agree.')
        for i in range(len(predicting_sample_keys)):
            sample_key = predicting_sample_keys[i]
            if predicted_label[i] > 0:
                self.dat.setdefault(target_label, set())
                self.dat[target_label].add(sample_key)
        return self

    def convert_to_original_index(self, index_mapping_rel):
        for k in self.dat.keys():
            new_index = set()
            for index in self.dat[k]:
                new_index.add(index_mapping_rel[index])
            self.dat[k] = new_index
        return self

    def evaluation(self, fact, predicted_labels):
        return macro_precision_and_recall(fact, self.dat, predicted_labels)
