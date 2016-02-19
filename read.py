# coding=utf-8
from array import array
from memory_profiler import profile
from scipy.sparse import csr_matrix
from data_analysis.labels import occurrence


class Sample:
    def __init__(self):
        self.y = list()
        self.x = None
        self.class_cnt = -1
        self._element = array('f')
        self._row_indptr = array('I')
        self._row_indptr.append(0)
        self._col_index = array('I')

    def __len__(self):
        return len(self.y)

    def read(self, data_file_path):
        with open(data_file_path) as f:
            for line_no, line in enumerate(f.__iter__()):
                self._read_one_line(line)
        self._convert_x_to_csr()
        return self

    def extract_and_update(self):
        test_instances, common_labels_cnt = self._select_instances()

        test_smp = Sample()
        train_smp = Sample()

        row_cnt, feature_dimension = self.x.shape

        for row_no in range(len(self.y)):
            begin, end = self._row_indptr[row_no], self._row_indptr[row_no + 1]
            if row_no in test_instances:
                test_smp.y.append(self.y[row_no])
                test_smp._element.extend(self._element[begin: end])
                test_smp._col_index.extend(self._col_index[begin: end])
                test_smp._row_indptr.append(test_smp._row_indptr[-1] + (end - begin))
            else:
                train_smp.y.append(self.y[row_no])
                train_smp._element.extend(self._element[begin: end])
                train_smp._col_index.extend(self._col_index[begin: end])
                train_smp._row_indptr.append(train_smp._row_indptr[-1] + (end - begin))

        test_smp.x = csr_matrix((test_smp._element, test_smp._col_index, test_smp._row_indptr),
                                shape=(len(test_instances), feature_dimension), dtype='float')
        test_smp.class_cnt = self.class_cnt
        train_smp.x = csr_matrix((train_smp._element, train_smp._col_index, train_smp._row_indptr),
                                 shape=(len(self.y) - len(test_instances), feature_dimension), dtype='float')
        train_smp.class_cnt = self.class_cnt
        return train_smp, test_smp, common_labels_cnt

    def _read_one_line(self, line):
        multi_label, instance = line.strip().split(' ', 1)

        all_labels = array('I', [int(l) for l in multi_label.split(',')])
        self.y.append(all_labels)
        for label in all_labels:
            self.class_cnt = label + 1 if label > self.class_cnt else self.class_cnt

        all_features = instance.split(' ')
        self._row_indptr.append(len(all_features) + self._row_indptr[-1])
        for feature in all_features:
            column, element = feature.split(':')
            column = int(column)
            self._element.append(float(element))
            self._col_index.append(column)
        return self

    def _convert_x_to_csr(self):
        self.x = csr_matrix((self._element, self._col_index, self._row_indptr),
                            shape=(len(self._row_indptr) - 1, max(self._col_index) + 1), dtype='float')

    def _select_instances(self):
        test_instances = set()
        label_occurrence = occurrence(self.y)
        common_label_cnt = 0.
        for label, instance_of_label in label_occurrence.items():
            if len(instance_of_label) > 2:
                test_instances.add(instance_of_label.pop())
                test_instances.add(instance_of_label.pop())
                common_label_cnt += 1
        return test_instances, common_label_cnt


if __name__ == '__main__':
    smp = Sample()
    smp.read('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt')
    print smp.y
    print smp.x
    print smp.class_cnt
    tr_smp, te_smp = smp.extract_and_update()
    print "================"
    print tr_smp.y
    print tr_smp.x
    print tr_smp.class_cnt
    print "==============="
    print te_smp.y
    print te_smp.x
    print te_smp.class_cnt
