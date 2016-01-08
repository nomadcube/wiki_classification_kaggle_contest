# coding=utf-8
from sklearn.naive_bayes import MultinomialNB
from ndarray_sorting import top
import numpy as np


class MultiClassMultinomialNB(MultinomialNB):
    def multi_predict(self, X, k):
        res = list()
        jll = self._joint_log_likelihood(X)
        top_feature_index = top(jll, k)
        top_k_features = np.array(self.classes_[top_feature_index]).reshape((-1, k))
        for row_id in range(top_k_features.shape[0]):
            res.append(','.join(top_k_features[row_id]))
        return res
