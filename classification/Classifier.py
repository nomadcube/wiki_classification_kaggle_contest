import re
import liblinearutil
from model_evaluation.evaluation import macro_metric
from Data.TrainData import TrainData


class Classifier:
    """Represent one classifier, with certain model and learning algorithm."""

    def __init__(self, train_y, train_x, prediction_path):
        self.y = train_y
        self.x = train_x
        self.y_remapping_rel = train.y.remapping_relation
        self.model = None
        self.evaluation_metric = tuple()
        self._prediction_path = prediction_path

    def learn(self):
        y = [self.y[key] for key in self.y.keys() if key in self.x.keys()]
        x = [dict(v) for v in self.x.values()]
        self.model = liblinearutil.train(y, x, '-s 0 -c 1')

    def predict(self):
        reverse_y_remapping_rel = {v: k for (k, v) in self.y_remapping_rel.items()}
        test_y = [self.y[key] for key in self.y.keys() if key in self.x.keys()]
        test_x = [dict(v) for v in self.x.values()]
        predicted_y = liblinearutil.predict(test_y, test_x, self.model)[0]
        with open(self._prediction_path, 'w') as predict_data:
            for index, each_predicted_y in enumerate(predicted_y):
                predict_data.write(str(index) + ',' +
                                   re.sub(',', ' ', str(reverse_y_remapping_rel[each_predicted_y])) + '\n')
                predict_data.flush()

    def evaluation(self):
        self.evaluation_metric = macro_metric(self.original_y, self._prediction_path)
