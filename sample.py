# coding=utf-8
import math

from data_analysis.labels import occurrence


class Sample:
    def __init__(self):
        self.y = list()
        self.x = None

    def __len__(self):
        return len(self.y)

    def train_cv_split(self):
        test_instances = self._select_instances()
        test_instances = list(test_instances)
        train_instances = list(set(range(len(self.y))).difference(test_instances))

        test_smp = Sample()
        train_smp = Sample()

        test_smp.y = [self.y[i] for i in test_instances]
        test_smp.x = self.x[test_instances, :]

        train_smp.y = [self.y[i] for i in train_instances]
        train_smp.x = self.x[train_instances, :]

        return train_smp, test_smp

    def _select_instances(self):
        test_instances = set()
        label_occurrence = occurrence(self.y)
        for label, instance_of_label in label_occurrence.items():
            if len(instance_of_label) > 1:
                num_in_cv = int(math.ceil(len(instance_of_label) * 0.2))
                if num_in_cv > 0:
                    for i in range(num_in_cv):
                        test_instances.add(instance_of_label.pop())
        return test_instances

    def convert_to_binary_class(self, target_label):
        binary_y = list()
        for each_label in self.y:
            new_label = 1 if each_label == target_label else 0
            binary_y.append(new_label)
        self.y = binary_y
