# coding=utf-8
from preprocessing.transforming import convert_y_to_coo
from array import array
from numpy.ma import masked_values


def macro_precision_recall(y, predicted_y, total_label_cnt, common_labels_cnt):
    # print predicted_y
    y_mat = convert_y_to_coo(y, element_dtype='float').tocsr()  # y_mat/pred_mat: 类别数x样本数
    pred_mat = convert_y_to_coo(predicted_y, element_dtype='float', total_label_cnt=y_mat.shape[0]).tocsr()

    inter_mat = y_mat.multiply(pred_mat)  # 可能有些类别在训练集上有而在测试集上没有，这会导致pred_mat的列数可能比y_mat大
    y_pos = masked_values(y_mat.sum(axis=1), 0.)
    pred_pos = masked_values(y_mat.sum(axis=1), 0.)
    true_positive = inter_mat.sum(axis=1)
    precision = true_positive / y_pos
    recall = true_positive / pred_pos
    return precision.sum() / common_labels_cnt, recall.sum() / common_labels_cnt


if __name__ == '__main__':
    test_y = [array('I', [65L, 66L, 67L, 68L, 69L]), array('I', [15L, 66L, 17L, 18L])]
    test_predicted_y = [[0], [15]]
    print(macro_precision_recall(test_y, test_predicted_y, 70, 8)[0])
    print(macro_precision_recall(test_y, test_predicted_y, 70, 8)[1])
