from Classifier import Classifier
from Data.TrainData import TrainData
from Data.generate_train_sample import generate_train_sample


generate_train_sample(100, 100)
TR = TrainData('/Users/wumengling/PycharmProjects/kaggle/input_data/train_sample.csv')
c1 = Classifier(TR, '/Users/wumengling/PycharmProjects/kaggle/output_data/model_fitting_predict.txt', 0.1)
c1.learn()
c1.predict()
c1.evaluation()
print(c1.evaluation_metric)

