import re
import liblinearutil
from model_evaluation.evaluation import macro_metric
from Data.TrainData import TrainData


class Classifier:
    """Represent one classifier, with certain model and learning algorithm."""

    def __init__(self, train, test, prediction_path):
        if not isinstance(train, TrainData):
            raise TypeError()
        if not isinstance(test, TrainData):
            raise TypeError()
        self._train = train
        self._test = test
        self.model = None
        self.evaluation_metric = tuple()
        self._prediction_path = prediction_path

    def learn(self):
        y = self._train.y.remap().reampped_data.values()
        x = self._train.x.dim_reduction(-1).dimension_reduction_data.values()
        self.model = liblinearutil.train(y, x, '-s 0 -c 1')

    def predict(self):
        y = self._test.y.remap().reampped_data.values()
        x = self._test.x.dim_reduction(-1).dimension_reduction_data.values()
        predicted_y = liblinearutil.predict(y, x, self.model)[0]
        with open(self._prediction_path, 'w') as predict_data:
            for index, each_predicted_y in enumerate(predicted_y):
                predict_data.write(str(index) + ',' +
                                   re.sub(', ', ' ', str(self._train.y._remapping_relation[each_predicted_y]) + '\n'))
                predict_data.flush()

    def evaluation(self):
        self.evaluation_metric = macro_metric(self._test.y.data, self._prediction_path)
