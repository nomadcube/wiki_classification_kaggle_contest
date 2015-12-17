import io


class TrainData:
    """Represent training data of machine learning."""

    def __init__(self, train_data_path):
        self.dat = io.open(train_data_path, 'r').readlines()
        self.instance_count = self._get_instance_count()
        self.y, self.x = self._get_y_x()
        self.label = self._get_label_set()
        self.y_mapping_relation = self._get_y_mapping_relation()
        self.feature_set = self._get_feature_set()

    def y_remapped(self):
        y_remapped_res = dict()
        for instance_index in self.y.keys():
            y_remapped_res[instance_index] = self.y_mapping_relation[', '.join([str(label)
                                                                               for label in self.y[instance_index]])]
        return y_remapped_res

    def _get_y_mapping_relation(self):
        res = dict()
        original_y_set = list()
        for label_set in self.y.values():
            tmp_label_str = ', '.join([str(label) for label in label_set])
            if tmp_label_str in original_y_set:
                continue
            original_y_set.append(tmp_label_str)
        for remapped_y, original_y in enumerate(original_y_set):
            res[original_y] = remapped_y
        return res

    # ------- methods used for initiation ---------
    def _get_instance_count(self):
        return len(self.dat)

    def _get_y_x(self):
        y_res = dict()
        x_res = dict()
        for index, line in enumerate(self.dat):
            y_res[index] = list()
            x_res[index] = dict()
            for column in line.strip().split(' '):
                if ':' not in column:
                    y_res[index].append(int(column.strip(',')))
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
    print(tr.y_mapping_relation)
