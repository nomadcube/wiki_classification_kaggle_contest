from collections import namedtuple


def true_id_per_label(test_data_path):
    """Convert test data with id-true_label into a map with label as its key and id set as value."""
    true_id = dict()
    with open(test_data_path, 'r') as true_label:
        for document_id, line in enumerate(true_label.readlines()):
            element_list = line.strip().split(' ')
            for label in element_list:
                if ':' in label:
                    continue
                label = label.strip(',')
                true_id.setdefault(label, set())
                true_id[label].add(document_id)
    return true_id


def confusion_matrix_per_label(predict_data_path, true_label_id):
    """Calculate each label's evaluation metric."""
    measure = namedtuple('measure', 'true_pos false_pos true_neg false_neg')
    confusion_matrix = dict()
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
            confusion_matrix.setdefault(label, measure(true_pos, false_pos, true_neg, false_neg))
    return confusion_matrix


def macro_metric(test_data_path, predict_data_path):
    """Calculate macro metric for multi-class classification."""
    tmp_macro_precision = 0.0
    tmp_macro_recall = 0.0
    true_id_map = true_id_per_label(test_data_path)
    confusion_mat = confusion_matrix_per_label(predict_data_path, true_id_map)
    for label in confusion_mat.keys():
        pos_count = confusion_mat[label].true_pos + confusion_mat[label].false_pos
        true_count = confusion_mat[label].true_pos + confusion_mat[label].false_neg
        # todo: skip the zero division problem may cause wrong.
        if (pos_count == 0) or (true_count == 0):
            continue
        each_precision = confusion_mat[label].true_pos / pos_count
        each_recall = confusion_mat[label].true_pos / true_count
        tmp_macro_precision += each_precision
        tmp_macro_recall += each_recall
    return tmp_macro_precision / len(confusion_mat), tmp_macro_recall / len(confusion_mat)
