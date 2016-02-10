from scipy.sparse import csr_matrix, csc_matrix, lil_matrix


def predict_label(x, w, b):
    labels = list()

    for sample_no in xrange(len(x.data)):
        sample_indices_data = {x.rows[sample_no][i]: x.data[sample_no][i] for i in xrange(len(x.data[sample_no]))}
        max_class = 0
        max_score = -1e30
        for label_no in xrange(len(w.data)):
            label_indices_data = {w.rows[label_no][i]: w.data[label_no][i] for i in xrange(len(w.data[label_no]))}
            if len(label_indices_data) == 0:
                continue
            sample_class_score = b[label_no]
            if sample_class_score == -float("inf") or len(
                    set(sample_indices_data.keys()).difference(set(label_indices_data.keys()))) > 0:
                continue
            # print sample_no, label_no
            for feature in set(sample_indices_data.keys()).intersection(set(label_indices_data.keys())):
                sample_class_score += sample_indices_data[feature] * label_indices_data[feature]
            if sample_class_score > max_score:
                max_score = sample_class_score
                max_class = label_no
        labels.append([max_class])
    return labels


if __name__ == '__main__':
    import random
    import numpy as np
    import fit_multi_label_mnb
    import cProfile, pstats, StringIO

    pr = cProfile.Profile()
    pr.enable()

    test_x = np.random.randint(1, 5, size=(90, 100))
    test_y = np.array([[random.randint(0, 9)] for i in range(90)])
    test_w = fit_multi_label_mnb.fit(test_y, csr_matrix(test_x))[1]
    test_b = fit_multi_label_mnb.fit(test_y, csr_matrix(test_x))[0]
    print predict_label(lil_matrix(test_x), test_w.tolil(), test_b)
    print test_y
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()
