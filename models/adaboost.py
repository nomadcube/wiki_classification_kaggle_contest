# coding=utf-8
import numpy as np
import math


class AdaBoost:
    def __init__(self, classifier_type, num_classifiers):
        self._classifier_type = classifier_type
        self._num_classifiers = 3
        self._models = list()
        self._coefs = list()

    def fit(self, y, x, ):
        num_smp = x.shape[0]
        smp_weight = np.ones(num_smp) / num_smp
        for _ in xrange(self._num_classifiers):
            current_model, current_coef, smp_weight = self.one_train(y, x, smp_weight)
            self._models.append(current_model)
            self._coefs.append(current_coef)
        return self

    def predict(self, x):
        prob = self.post_prob(x)
        return np.array(np.argmax(prob, axis=1).ravel())

    def post_prob(self, x):
        prob = np.zeros((x.shape[0], 2))
        for i, model in enumerate(self._models):
            current_post_prob = model.post_prob(x)
            prob += current_post_prob * self._coefs[i]
        return prob

    def one_train(self, y, x, smp_weight):
        model = self._classifier_type()
        model.fit(y, x, smp_weight)
        prediction = model.predict(x)
        error_rate = (smp_weight * np.array(prediction != y, dtype='int8')).sum()
        weak_classifier_coef = math.log((1 - error_rate) / error_rate) / 2.0
        update_coef = np.exp((-weak_classifier_coef) * np.array(prediction == y, dtype='int8'))
        smp_weight *= update_coef
        smp_weight /= smp_weight.sum()
        return model, weak_classifier_coef, smp_weight
