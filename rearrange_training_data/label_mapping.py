def label(train_data_path):
    """Distinct label generator."""
    res = set()
    with open(train_data_path, 'r') as train_data:
        for line in train_data.readlines()[1:]:
            label_list = list()
            for column in line.strip().split(' '):
                if ":" not in column:
                    label_list.append(column.strip(','))
            res.add(','.join(label_list))
    return res


def label_mapping(label_set):
    """Map real label vector to its index."""
    res = dict()
    for mapped_label, label_string in enumerate(label_set):
        print(mapped_label, label_string)
        res[label_string] = mapped_label
    return res


if __name__ == '__main__':
    lab = label('/Users/wumengling/kaggle/unit_test_data/sample.txt')
    print(lab)
    print(label_mapping(lab))
