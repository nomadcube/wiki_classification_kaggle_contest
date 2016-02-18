from preprocessing.transforming import convert_y_to_csr
from array import array


def macro_precision_recall(y, predicted_y):
    precision = array('f')
    recall = array('f')

    y_mat = convert_y_to_csr(y, element_dtype='float').transpose().tolil()
    pred_mat = convert_y_to_csr(predicted_y, element_dtype='float', max_n_dim=y_mat.shape[0]).transpose().tolil()

    for row_no in range(y_mat.shape[0]):
        if y_mat[row_no].nnz == 0:
            continue
        inter_mat = y_mat[row_no].multiply(pred_mat[row_no])
        true_positive = inter_mat.data.sum() if inter_mat.nnz > 0 else 0.
        y_pos = sum(y_mat[row_no].data[0])
        pred_pos = sum(pred_mat[row_no].data[0])
        precision.append(true_positive / y_pos) if y_pos > 0. else precision.append(0.)
        recall.append(true_positive / pred_pos) if pred_pos > 0. else recall.append(0.)
    return sum(precision) / len(precision), sum(recall) / len(recall)


if __name__ == '__main__':
    test_y = [array('I', [65L, 66L, 67L, 68L, 69L]), array('I', [15L, 66L, 17L, 18L])]
    test_predicted_y = [[0], [15]]
    print(macro_precision_recall(test_y, test_predicted_y)[0])
    print(macro_precision_recall(test_y, test_predicted_y)[1])
