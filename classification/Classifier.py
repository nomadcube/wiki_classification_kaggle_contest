# coding=utf-8
import re
import liblinearutil
from perceptron import perceptron


class Classifier:
    """Represent one classifier, with certain model and learning algorithm."""
    # todo: 事实上，这个类似乎是不需要的，所做的都是main.py该做的事情，即它要做的事情用函数就可以完成，不需要有数据成员

    def __init__(self, train_y, train_x, y_remap_rel):
        """Initiate a Classifier object with train data and remapping relationship."""
        self.y = train_y
        self.x = train_x
        self.y_remapping_rel = y_remap_rel  # todo: 因此也不需要有这个数据成员
        self.model = None

    def learn(self, algorithm):
        """Learn linear model using liblinear."""
        if algorithm == 'liblinear':
            self.model = liblinearutil.train(self.y, self.x, '-s 0 -c 1')
        if algorithm == 'perceptron':
            self.model = perceptron.learning(100, 0.3, self.x, self.y, len(self.x.values()))
        return self

    def _predict(self, y_to_predict, x_to_predict):
        """Make prediction on y_to_predict and x_to_predict."""
        # todo: 只应该做预测这么一件事情，不需要利用y_remapping_rel将预测结果转回去
        predicted_y, total_accuracy = liblinearutil.predict(y_to_predict, x_to_predict, self.model)[:2]
        if len(self.y_remapping_rel) > 0:
            predict_res = list()
            reverse_y_remapping_rel = {v: k for (k, v) in self.y_remapping_rel.items()}
            for each_predicted_y in predicted_y:
                predict_res.append(re.sub(',', ' ', str(reverse_y_remapping_rel[each_predicted_y])))
            return predict_res, total_accuracy
        else:
            return predicted_y, total_accuracy

    def evaluation(self, y_to_evaluate, x_to_evaluate):
        """Make evaluation on y and x."""
        # todo: 不应该是模型的成员方法，可以另写成一个独立的函数
        return self._predict(y_to_evaluate, x_to_evaluate)[1]
        # original_y = [self.y_remapping_rel[mapped_y] for mapped_y in y_to_evaluate]
        # return macro_metric(original_y, predicted_y)
