import liblinearutil

from data_processing.libsvm_train_data import training_label, dimension_reduction_instance
from data_processing.tf_idf import term_frequency, log_inverse_doc_frequency


def training(y_train, x_train):
    m = liblinearutil.train(y_train, x_train, '-s 0 -c 1')
    return m


def predicting(y_test, x_test, model, predict_path):
    """Make prediction on [y_test, x_test] with model generated in training."""
    p_label, p_acc, p_val = liblinearutil.predict(y_test, x_test, model)
    with open(predict_path, 'w') as predict_data:
        for index, label in enumerate(p_label):
            predict_data.write(str(index) + ',' + str(int(label)) + '\n')
            predict_data.flush()


if __name__ == '__main__':
    train_data_path = '/Users/wumengling/kaggle/unit_test_data/sample.txt'
    predict_data_path = '/Users/wumengling/kaggle/unit_test_data/fitting_predict.txt'
    y = [i for i in training_label(train_data_path)]
    tf = term_frequency(train_data_path)
    idf = log_inverse_doc_frequency(train_data_path)
    x = [i for i in dimension_reduction_instance(tf, idf, 0)]
    lr_model = training(y, x)
    predicting(y, x, lr_model, predict_data_path)
