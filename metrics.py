# coding=utf-8
import numpy as np

from array import array
from numpy.ma import masked_values
from collections import namedtuple

from preprocessing.transforming import convert_y_to_csr


def get_evaluation_metrics(y, predicted_y, mat_shape, metrics_denominator=None):
    """
    :param y: list of array, list长度为样本量，array长度为各自包含的label数
    :param predicted_y: list of array, list长度为样本量，array长度为各自包含的预测label数
    :param metrics_denominator: float/int, 计算平均precision和recall时的分母，默认为None, 这时用的是y中的label数
    :return: 由y和predicted_y算出来的平均precision, recall和f-score

    ### train和cv
    y_mat和pred_mat所包含label的关系：
    1. 若没有限制w所能包含的最大label数max_num_label，则{y_mat}是{pred_mat}的子集
    2. 否则{y_mat}和{pred_mat}都只能包含最多max_num_label个label
    因此在算平均precision和recall时，分母应该是min(max_num_label, y_mat所包含的label个数)
    另外，由于y_mat和pred_mat的行号都是原label，而由于pred_mat一定是大于等于y_mat的，因此可以将它们的行数即shape[0]都设为y_mat中的最大label.

    ### test和cv
    y_mat和pred_mat所包含label的关系：
    各自和对方都有相交和不相交的部分
    在算平均precision和recall时，以y_mat为主，因此分母应该是y_mat所包含的label个数
    另外，由于y_mat和pred_mat的行号都是原label，而由于pred_mat不一定大于等于y_mat的，因此需要将行数设为smp中的全局最大label.

    结论：
    1. y_mat和pred_mat的total_label_cnt都要设为smp中的最大label
    2. 对于train和cv, 将算平均precision和recall时的分母设为min(max_num_label, y_mat所包含的label个数)
    3. 对于test, 将算平均precision和recall时的分母设为y_mat所包含的label个数


    """
    # 若某个类别只在y_mat中出现或只在pred_mat中出现，对应的precision和recall都设置成0
    evaluation_metrics = namedtuple('evaluation_metrics', 'mpr mre f_score')

    y_mat = convert_y_to_csr(y, element_dtype='float', total_label_cnt=mat_shape + 1)
    pred_mat = convert_y_to_csr(predicted_y, element_dtype='float', total_label_cnt=mat_shape + 1)

    non_empty_rows_no = np.diff(y_mat.indptr)
    num_y_label = np.where(non_empty_rows_no > 0)[0].shape[0]

    inter_mat = y_mat.multiply(pred_mat)
    y_pos = masked_values(y_mat.sum(axis=1), 0.)
    pred_pos = masked_values(pred_mat.sum(axis=1), 0.)
    true_positive = inter_mat.sum(axis=1)
    precision = true_positive / y_pos
    recall = true_positive / pred_pos
    metrics_denominator = min(metrics_denominator, num_y_label) if metrics_denominator is not None else num_y_label
    m_precision = precision.sum() / metrics_denominator
    m_recall = recall.sum() / metrics_denominator
    f_score = 1. / (1. / m_precision + 1. / m_recall) if m_precision != 0. and m_recall != 0. else float("inf")
    return evaluation_metrics(m_precision, m_recall, f_score)


if __name__ == '__main__':
    test_y = [array('I', [65L, 66L, 67L, 68L, 69L]), array('I', [15L, 66L, 17L, 18L])]
    test_predicted_y = [[0], [15]]
    print(get_evaluation_metrics(test_y, test_predicted_y, 70, 8)[0])
    print(get_evaluation_metrics(test_y, test_predicted_y, 70, 8)[1])
