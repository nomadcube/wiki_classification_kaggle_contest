# coding=utf-8
import numpy as np


class Node:
    """
    代表一个树节点
    """

    def __init__(self, sample_index_subset):
        self._sample_index_subset = sample_index_subset

    def entropy(self, y):
        """
        计算当前树节点所代表的样本子集的熵

        :param y: array, 全量样本集对应的目标变量
        :return: float
        """
        counts = list()
        y_subset = y[self._sample_index_subset]
        label_cnt = float(len(y_subset))
        y_subset.sort()
        difference_index = np.where(np.diff(y_subset) > 0)[0]
        counts.append(difference_index[0] + 1.0)
        counts.extend(list(np.diff(difference_index)))
        counts.append(label_cnt - sum(counts))
        counts = np.array(counts) / label_cnt
        return (-1.0) * (np.log(counts) * counts).sum()

    def split(self, feature_index, x):
        """
        根据所选取的特征及其取值，将当前节点分裂成若干个子结点

        :param feature_index: int, 所选取特征的索引号
        :param x: matrix, 自变量数据
        :return: list of Node, 由若干个子结点组成的列表
        """
        children = list()
        one_column_x = x[:, feature_index]
        distinct_values = list(np.unique(one_column_x))
        for each_value in distinct_values:
            children.append(Node(np.where(one_column_x == each_value)))
        return children


if __name__ == '__main__':
    all_y = np.array([0, 1, 0, 0, 1, 2])
    node = Node([0, 1, 4, 2, 3])
    print node.entropy(all_y)
