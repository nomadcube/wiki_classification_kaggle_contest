import Classifier

from data_processing.TrainData import TrainData


def test_classifier():
    c1 = Classifier.Classifier(TrainData('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt'),
                               '/Users/wumengling/PycharmProjects/kaggle/unit_test_data/model_fitting_predict.txt')
    c1.learn()
    c1.make_prediction()
    c1.make_evaluation()
    assert c1.evaluation_metric == (1.0, 1.0)
