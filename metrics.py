# coding=utf-8
from preprocessing.transforming import convert_y_to_csr
from array import array


def macro_precision_recall(y, predicted_y, common_labels_cnt):
    precision = array('f')
    recall = array('f')

    y_mat = convert_y_to_csr(y, element_dtype='float').transpose().tolil()  # y_mat/pred_mat: 类别数x样本数
    pred_mat = convert_y_to_csr(predicted_y, element_dtype='float', total_label_cnt=y_mat.shape[0]).transpose().tolil()

    for row_no in range(y_mat.shape[0]):
        one_y_mat = y_mat[row_no]
        if one_y_mat.nnz == 0:
            continue
        one_pred_mat = pred_mat[row_no]
        inter_mat = one_y_mat.multiply(one_pred_mat)
        true_positive = inter_mat.data.sum() if inter_mat.nnz > 0 else 0.
        y_pos = sum(one_y_mat.data[0])
        pred_pos = sum(one_pred_mat.data[0])
        precision.append(true_positive / y_pos) if y_pos > 0. else precision.append(0.)
        recall.append(true_positive / pred_pos) if pred_pos > 0. else recall.append(0.)
    return sum(precision) / common_labels_cnt, sum(recall) / common_labels_cnt


if __name__ == '__main__':
    test_y = [array('I', [65L, 66L, 67L, 68L, 69L]), array('I', [15L, 66L, 17L, 18L])]
    test_predicted_y = [[0], [15]]
    print(macro_precision_recall(test_y, test_predicted_y)[0])
    print(macro_precision_recall(test_y, test_predicted_y)[1])
