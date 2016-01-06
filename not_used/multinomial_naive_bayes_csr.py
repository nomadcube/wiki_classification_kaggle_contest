import numpy as np

from scipy.sparse import csr_matrix
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB


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
    iris_arr = iris.data
    iris_csr = construct_csr_from_array(iris_arr)
    print(iris_csr)
    nb = MultinomialNB()
    nb_classifier = nb.fit(iris_csr, iris.target)
    y_predict = nb_classifier.predict(iris_csr)
    print(y_predict)
    print(iris.target)
    print((iris.target != y_predict).sum())
