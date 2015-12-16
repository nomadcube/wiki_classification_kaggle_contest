from collections import namedtuple


def real_label_id_map(test_data_path):
    """Convert test data with id-true_label into a map with label as its key and id set as value."""
    label_id = dict()
    with open(test_data_path, 'r') as true_label:
        for document_id, line in enumerate(true_label.readlines()[1:]):
            element_list = line.strip().split(' ')
            for label in element_list:
                if ':' in label:
                    continue
                label = label.strip(',')
                label_id.setdefault(label, set())
                label_id[label].add(document_id)
    return label_id


def each_label_metric(predict_data_path, true_label_id):
    """Calculate each label's evaluation metric."""
    measure = namedtuple('measure', 'true_pos false_pos true_neg false_neg')
    label_measure = dict()
    with open(predict_data_path, 'r') as predict_data:
        all_line = predict_data.readlines()[1:]
        for label in true_label_id.keys():
            true_pos, false_pos, true_neg, false_neg = [0.0] * 4
            for line in all_line:
                index = int(line.split(',')[0])
                predict_label_set = set(line.strip().split(',')[1].split(' '))
                if label in predict_label_set:
                    if index in true_label_id[label]:
                        true_pos += 1
                    else:
                        false_pos += 1
                else:
                    if index in true_label_id[label]:
                        false_neg += 1
                    else:
                        true_neg += 1
            label_measure.setdefault(label, measure(true_pos, false_pos, true_neg, false_neg))
    return label_measure


def macro_metric(each_label_measure):
    """Calculate macro metric for multi-class classification."""
    tmp_macro_precision = 0.0
    tmp_macro_recall = 0.0
    for label in each_label_measure.keys():
        pos_count = each_label_measure[label].true_pos + each_label_measure[label].false_pos
        true_count = each_label_measure[label].true_pos + each_label_measure[label].false_neg
        # todo: skip the zero division problem may cause wrong.
        if (pos_count == 0) or (true_count == 0):
            continue
        each_precision = each_label_measure[label].true_pos / pos_count
        each_recall = each_label_measure[label].true_pos / true_count
        tmp_macro_precision += each_precision
        tmp_macro_recall += each_recall
    return tmp_macro_precision / len(each_label_measure), tmp_macro_recall / len(each_label_measure)


if __name__ == '__main__':
    r_label_id_map = real_label_id_map('/Users/wumengling/kaggle/unit_test_data/sample.txt')
    label_measure = each_label_metric('/Users/wumengling/kaggle/unit_test_data/predict.txt', r_label_id_map)
    print(r_label_id_map)
    print(label_measure)
    print(0 in r_label_id_map['416827'])
