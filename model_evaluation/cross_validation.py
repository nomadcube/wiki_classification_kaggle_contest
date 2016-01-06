from random import random


def split_train_test(x, y, train_proportion):
    if len(x) != len(y):
        raise ValueError('Length of x and y must agree.')
    train_x = list()
    train_y = list()
    test_x = list()
    test_y = list()
    for instance_id in range(len(x)):
        determine_number = random()
        if determine_number <= train_proportion:
            train_x.append(x[instance_id])
            train_y.append(y[instance_id])
        else:
            test_x.append(x[instance_id])
            test_y.append(y[instance_id])
    return train_y, train_x, test_y, test_x
