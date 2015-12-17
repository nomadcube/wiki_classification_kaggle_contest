import liblinearutil


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
