import time
import TrainData
from generate_train_sample import generate_train_sample


generate_train_sample(500000, 100)
train_data = TrainData.TrainData('/Users/wumengling/PycharmProjects/kaggle/input_data/train_sample.csv')
start_time = time.time()
train_data.x.dim_reduction(2.0)
print(time.time() - start_time)
