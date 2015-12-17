import io


class TrainData:
    """Represent training data of machine learning."""

    def __init__(self, train_data_path):
        self.dat = io.open(train_data_path, 'r').readlines()
        self.instance_count = self._get_instance_count()
        self.y, self.x = self._get_y_x()
        self.label = self._get_label_set()
        self.label_mapping_relation = self._label_mapping()
        self.feature_set = self._get_feature_set()

    def y_remapped(self):
        remapped_y = dict()
        for instance_index in self.y.keys():
            remapped_y[instance_index] = set()
            for label in self.y[instance_index]:
                remapped_y[instance_index].add(self.label_mapping_relation[label])
        return remapped_y

    def _label_mapping(self):
        mapping_res = dict()
        for mapped_label, original_label in enumerate(self.label):
            mapping_res[original_label] = mapped_label
        return mapping_res

    # ------- methods used for initiation ---------
    def _get_instance_count(self):
        return len(self.dat)

    def _get_y_x(self):
        y_res = dict()
        x_res = dict()
        for index, line in enumerate(self.dat):
            y_res[index] = set()
            x_res[index] = dict()
            for column in line.strip().split(' '):
                if ':' not in column:
                    y_res[index].add(int(column.strip(',')))
                else:
                    feat, val = column.split(':')
                    feat = int(feat)
                    val = float(val)
                    x_res[index][feat] = val
        return y_res, x_res

    def _get_label_set(self):
        label_set_res = set()
        for each_y in self.y.values():
            for label in each_y:
                label_set_res.add(label)
        return label_set_res

    def _get_feature_set(self):
        feature_set = set()
        for each_instance in self.x.values():
            for feature in each_instance.keys():
                feature_set.add(feature)
        return feature_set


if __name__ == '__main__':
    tr = TrainData('/Users/wumengling/kaggle/unit_test_data/sample.txt')
    print(tr.label_mapping_relation)
