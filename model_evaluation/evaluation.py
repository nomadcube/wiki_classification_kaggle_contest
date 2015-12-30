from collections import namedtuple


def generate_fact(y, index_mapping_rel):
    res = dict()
    for instance_index, label in y.items():
        res.setdefault(label, set())
        res[label].add(index_mapping_rel[instance_index])
    return res


def collect_whole_index(fact):
    res = set()
    for index in fact.values():
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


def macro_precision_and_recall(fact, prediction):
    macro_precision = 0.0
    macro_recall = 0.0
    whole_index = collect_whole_index(fact)
    for each_label in fact.keys():
        if (each_label not in fact.keys()) or (each_label not in prediction.keys()):
            continue
        confusion_mat = confusion_matrix(fact[each_label], prediction[each_label], whole_index)
        tmp_precision, tmp_recall = precision_and_recall(confusion_mat)
        macro_precision += tmp_precision
        macro_recall += tmp_recall
    return macro_precision / len(fact.keys()), macro_recall / len(fact.keys())


class Prediction:
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

    def evaluation(self, fact):
        return macro_precision_and_recall(fact, self.dat)
