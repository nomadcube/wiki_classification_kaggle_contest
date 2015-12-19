import io


class TrainData:
    """Represent training data of machine learning."""

    def __init__(self, train_data_path):
        self.dat = io.open(train_data_path, 'r').readlines()
        self.y, self.x = self._get_y_x()

    # ------- methods used for extract info about x and y ---------
    def instance_count(self):
        return len(self.dat)

    def label_set(self):
        label_set_res = set()
        for each_y in self.y.values():
            for label in each_y.split(','):
                label_set_res.add(int(label))
        return label_set_res

    def feature_set(self):
        feature_set = set()
        for each_instance in self.x.values():
            for feature in each_instance.keys():
                feature_set.add(feature)
        return feature_set

    def _get_y_x(self):
        y_res = dict()
        x_res = dict()
        for index, line in enumerate(self.dat):
            x_res[index] = dict()
            tmp_y, tmp_x = line.strip().split(' ', 1)
            y_res[index] = tmp_y
            for column in tmp_x.split(' '):
                feat, val = column.split(':')
                feat = int(feat)
                val = float(val)
                x_res[index][feat] = val
        return y_res, x_res

    # ------- methods used for remapping y as of type int ---------
    def y_remapped(self):
        y_remapped_res = dict()
        remapped_relation = self.y_mapping_relation()
        for instance_index in self.y.keys():
            y_remapped_res[instance_index] = remapped_relation[self.y[instance_index]]
        return y_remapped_res

    def y_mapping_relation(self):
        res = dict()
        remapped_y = 0
        for each_y in self.y.values():
            if each_y in res.keys():
                continue
            res[each_y] = remapped_y
            remapped_y += 1
        return res


if __name__ == '__main__':
    tr = TrainData('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt')
    print(len(tr.dat))
