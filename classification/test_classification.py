from Classifier import Classifier
from Data.TrainData import TrainData


def test_classifier():
    # todo: wtf, getting complicated and chaos now.
    TR = TrainData('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt')
    TE = TR
    c1 = Classifier(TR, TE, '/Users/wumengling/PycharmProjects/kaggle/unit_test_data/model_fitting_predict.txt')
    c1.learn()
    c1.predict()
    c1.evaluation()
    assert c1.evaluation_metric == (1.0, 1.0)
