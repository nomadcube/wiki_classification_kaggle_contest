import re
import liblinearutil
from model_evaluation.evaluation import macro_metric
from Data.TrainData import TrainData


class Classifier:
    """Represent one classifier, with certain model and learning algorithm."""

    def __init__(self, train, prediction_path, tf_idf_threshold=-1.0):
        if not isinstance(train, TrainData):
            raise TypeError()
        self.y = train.y.remapped_data
        self.original_y = train.y.data
        train.x.dim_reduction(tf_idf_threshold)
        self.x = dict(train.x.dim_reduction_data)
        self.y_remapping_rel = train.y.remapping_relation
        self.model = None
        self.evaluation_metric = tuple()
        self._prediction_path = prediction_path

    def learn(self):
        y = [self.y[key] for key in self.y.keys() if key in self.x.keys()]
        self.model = liblinearutil.train(y, self.x.values(), '-s 0 -c 1')

    def predict(self):
        reverse_y_remapping_rel = {v: k for (k, v) in self.y_remapping_rel.items()}
        predicted_y = liblinearutil.predict([self.y[key] for key in self.y.keys() if key in self.x.keys()], self.x.values(), self.model)[0]
        with open(self._prediction_path, 'w') as predict_data:
            for index, each_predicted_y in enumerate(predicted_y):
                predict_data.write(str(index) + ',' +
                                   re.sub(',', ' ', str(reverse_y_remapping_rel[each_predicted_y])) + '\n')
                predict_data.flush()

    def evaluation(self):
        self.evaluation_metric = macro_metric(self.original_y, self._prediction_path)
