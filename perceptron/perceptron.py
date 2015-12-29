from sparse_vector.sparse_vector import sparse_vector_dat_t, SparseVector


def if_wrong_discriminated(x, y, weight):
    """
    Given current weight, judge if the give sample [x, y] is wrong discriminated.

    :param weight: SparseVector
    :param x: SparseVector
    :param y: float
    """
    return x.inner_product(weight) * y <= 0


def update_weight(current_weight, learning_rate, negative_gradient):
    """
    Update weight with learning rate and the current gradient.

    :param current_weight: SparseVector
    :param learning_rate: float
    :param negative_gradient: SparseVector
    :return: SparseVector
    """
    return current_weight + negative_gradient.dot_multiplication(learning_rate)


def learning(max_iter_time, learning_rate, X, Y, feature_dimension):
    """
    Train a perceptron classifier with training data [X, Y].
    Stop iterating when reaching max_iter_time.

    :param max_iter_time: int
    :param learning_rate: float
    :param X: list
    :param Y: list
    :return: SparseVector
    """
    sample_size = len(X)
    init_weight = SparseVector(sparse_vector_dat_t({key + 1: 0.0 for key in range(feature_dimension)}))
    current_sample_index = 0
    while current_sample_index < sample_size:
        if if_wrong_discriminated(X[current_sample_index], Y[current_sample_index], init_weight):
            init_weight = update_weight(init_weight, learning_rate,
                                        X[current_sample_index].dot_multiplication(Y[current_sample_index]))
            current_sample_index = 0
        else:
            current_sample_index += 1
    return init_weight


def predict(weight, x):
    """
    Given the fitted model parameters weight, predict if sample x is negative of positive.

    :param weight: SparseVector
    :param x: SparseVector
    :return: float
    """
    decision_val = x.inner_product(weight)
    if decision_val > 0:
        return 1.0
    else:
        return -1.0


if __name__ == '__main__':
    X = [SparseVector(sparse_vector_dat_t({1: 3.0, 2: 3.0, 3: 1.0})),
         SparseVector(sparse_vector_dat_t({1: 4.0, 2: 3.0, 3: 1.0})),
         SparseVector(sparse_vector_dat_t({1: 1.0, 2: 1.0, 3: 1.0}))]
    Y = [1.0, 1.0, -1.0]
    final_w = learning(100, 1.0, X, Y, 3)
    print(dict(final_w.dat))
