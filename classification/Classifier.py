import re
import liblinearutil
from model_evaluation.evaluation import macro_metric


class Classifier:
    """Represent one classifier, with certain model and learning algorithm."""

    def __init__(self, train_y, train_x, y_remap_rel):
        """Initiate a Classifier object with train data and remapping relationship."""
        self.y = train_y
        self.x = train_x
        self.y_remapping_rel = y_remap_rel
        self.model = None

    def learn(self):
        """Learn linear model using liblinear."""
        self.model = liblinearutil.train(self.y, self.x, '-s 0 -c 1')
        return self

    def _predict(self, y_to_predict, x_to_predict):
        """Make prediction on y_to_predict and x_to_predict."""
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
        return self._predict(y_to_evaluate, x_to_evaluate)[1]
        # original_y = [self.y_remapping_rel[mapped_y] for mapped_y in y_to_evaluate]
        # return macro_metric(original_y, predicted_y)
