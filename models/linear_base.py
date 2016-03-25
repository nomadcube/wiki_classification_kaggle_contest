def vector_p_norm(w, p):
    return abs(pow(sum(np.power(w, p)), 1. / p))


def discrimination(one_w, one_x):
    return one_w * one_x.transpose()
