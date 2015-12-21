import os


train_data_path = '/Users/wumengling/PycharmProjects/kaggle/input_data/train_sample.csv'
test_data_path = '/Users/wumengling/PycharmProjects/kaggle/input_data/test_sample.csv'
prediction_path = '/Users/wumengling/PycharmProjects/kaggle/output_data/model_fitting_predict.txt'


def generate_train_sample(train_data_line=10, test_data_line=10):
    """Generate train and test sample from complete data."""
    split_train_cmd = "sed -n '1," + str(train_data_line) + \
                      "p' /Users/wumengling/PycharmProjects/kaggle/input_data/train.csv | cat >" + train_data_path
    split_test_cmd = "sed -n '" + str(train_data_line + 1) + "," + str(train_data_line + test_data_line) + \
                     "p' /Users/wumengling/PycharmProjects/kaggle/input_data/train.csv | cat >" + test_data_path
    os.system(split_train_cmd)
    os.system(split_test_cmd)
