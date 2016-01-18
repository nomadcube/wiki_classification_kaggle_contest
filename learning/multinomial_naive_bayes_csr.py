import numpy as np

from scipy.sparse import csr_matrix
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from learning import chunking
from sklearn.feature_extraction.text import HashingVectorizer


def construct_csr_from_array(arr):
    if not isinstance(arr, np.ndarray):
        raise TypeError()
    data = list()
    row_ind = list()
    col_ind = list()
    row_num, col_num = arr.shape
    for i in range(row_num):
        for j in range(col_num):
            if arr[i, j] != 0:
                data.append(arr[i, j])
                row_ind.append(i)
                col_ind.append(j)
    return csr_matrix((data, (row_ind, col_ind)), shape=arr.shape)


if __name__ == '__main__':
    iris = datasets.load_iris()
    # iris_arr = iris.data_processing
    # iris_csr = construct_csr_from_array(iris_arr)
    # nb = MultinomialNB()
    # nb_classifier = nb.fit(iris_csr, iris.target)
    # y_predict = nb_classifier.predict(iris_csr)
    # print(y_predict)
    # print(iris.target)
    # print((iris.target != y_predict).sum())
    # print(len(iris.target))
    # y = iris.target
    y = [[original_y, 3 - original_y] for original_y in iris.target]
    x = iris.data
    mnb = MultinomialNB()
    whole_classes = [0, 1, 2, 3]
    for chuck_y, chuck_x in chunking.iter_chuck(y.__iter__(), x.__iter__(), 15):
        csr_chuck_x = construct_csr_from_array(np.array(chuck_x))
        mnb.partial_fit(csr_chuck_x, chuck_y, classes=whole_classes)
    y_predict = mnb.predict(construct_csr_from_array(x))
    print(y_predict)
