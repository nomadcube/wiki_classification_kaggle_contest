# coding=utf-8
import numpy as np
import math

from mnb import LaplaceSmoothedMNB


class AdaBoost:
    def __init__(self, week_classifier):
        self._week_classifier = week_classifier
        self.classifiers = list()
        self.classifier_weight = list()

    def fit(self, y, x, iter_time):
        sample_weight = np.ones(x.shape[0]) / x.shape[0]
        for _ in xrange(iter_time):
            model, predicted_y = self._weak_model_and_prediction(y, x, sample_weight)
            self.classifiers.append(model)
            self.classifier_weight.append(self._weak_classifier_coef(y, predicted_y, sample_weight))
            sample_weight = self._update_sample_weight(y, predicted_y, sample_weight)
        return self

    def _weak_model_and_prediction(self, y, x, sample_weight):
        model = self._week_classifier()
        weighted_x = (sample_weight * x.transpose()).transpose()
        model.fit(weighted_x, y, sample_weight)
        predicted_y = model.predict(x)
        return model, predicted_y

    def predict(self, x):
        final_predicted_y = np.zeros(x.shape[0])
        for i, model in enumerate(self.classifiers):
            predicted_y = model.predict(x)
            final_predicted_y += predicted_y * self.classifier_weight[i]
        return [-1 if each_y < 0 else 1 for each_y in final_predicted_y]

    def _error_rate(self, y, predicted_y, sample_weight):
        return (sample_weight * np.array(predicted_y != y, dtype='int8')).sum()

    def _weak_classifier_coef(self, y, predicted_y, sample_weight):
        error_rate = self._error_rate(y, predicted_y, sample_weight)
        return math.log((1 - error_rate) / error_rate) / 2.0

    def _update_sample_weight(self, y, predicted_y, sample_weight):
        weak_classifier_coef = self._weak_classifier_coef(y, predicted_y, sample_weight)
        update_coef = np.exp((-weak_classifier_coef) * np.array(predicted_y == y, dtype='int8'))
        sample_weight *= update_coef
        sample_weight /= sample_weight.sum()
        return sample_weight


if __name__ == '__main__':
    from sklearn.naive_bayes import MultinomialNB
    from scipy.sparse import csr_matrix

    sample_size = 10
    feature_size = 3
    y = np.array([[0] if i == -1 else[1] for i in [1, 1, -1, 1, 1, 1, -1, -1, -1, 1]])
    x = csr_matrix(np.random.randint(1, 4, (sample_size, feature_size)))

    mnb = LaplaceSmoothedMNB()
    mnb.fit(y, x)
    predicted_y = mnb.predict(x, 1)

    print np.array(predicted_y != y, dtype='int8').sum() / float(len(y))

    print y
    print predicted_y
    #
    # ada_boost = AdaBoost(LaplaceSmoothedMNB)
    # ada_boost.fit(y, x, 100)
    # predicted_y = ada_boost.predict(x)
    #
    # print np.array(predicted_y != y, dtype='int8').sum() / float(len(y))
    #
    # print y
    # print predicted_y
