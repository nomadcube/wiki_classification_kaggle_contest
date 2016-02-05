from scipy.sparse import csr_matrix, csc_matrix


def predict_label(x, w, b):
    if not isinstance(x, csr_matrix):
        raise TypeError()
    if not isinstance(w, csc_matrix):
        raise TypeError()

    x_zip = [(x.data[i], x.indices[i]) for i in range(len(x.data))]
    w_zip = [(w.indices[i], w.data[i]) for i in range(len(w.data))]

    labels = list()
    for sample_no in xrange(len(x.indptr) - 1):
        max_class = 0
        max_score = -1e30
        sample_zip = {v: k for (k, v) in x_zip[x.indptr[sample_no]: x.indptr[sample_no + 1]]}
        if len(sample_zip) > 0:
            for class_no in xrange(len(w.indptr) - 1):
                class_zip = {k: v for (k, v) in w_zip[w.indptr[class_no]: w.indptr[class_no + 1]]}
                if len(class_zip) > 0:
                    sample_class_score = b[class_no]
                    if len(set(sample_zip.keys()).difference(set(class_zip.keys()))) > 0:
                        continue
                    else:
                        for feature_no in sample_zip.keys():
                            sample_class_score += sample_zip[feature_no] * class_zip[feature_no]
                    if sample_class_score > max_score:
                        max_score = sample_class_score
                        max_class = class_no
        labels.append([max_class])
    return labels


if __name__ == '__main__':
    import random
    import numpy as np
    import fit_multi_label_mnb
    import cProfile, pstats, StringIO

    pr = cProfile.Profile()
    pr.enable()

    x = np.random.randint(1, 5, size=(90, 100))
    y = np.array([[random.randint(1, 10)] for i in range(90)])
    w = fit_multi_label_mnb.fit(y, csr_matrix(x))[1]
    b = fit_multi_label_mnb.fit(y, csr_matrix(x))[0]
    print predict_label(csr_matrix(x), w.transpose().tocsc(), b)
    print y
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()
