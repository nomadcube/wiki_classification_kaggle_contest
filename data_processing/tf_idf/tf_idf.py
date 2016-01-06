from math import log


def idf(x):
    res = dict()
    for term_freq in x.values():
        for each_term in term_freq.keys():
            res.setdefault(each_term, 0.0)
            res[each_term] += 1
    return res


def tf_idf(x, threshold):
    new_x = dict()
    idf_res = idf(x)
    size = float(len(x))
    for index, term_freq in x.items():
        new_x[index] = dict()
        freq_sum = float(sum(term_freq.values()))
        for term in term_freq.keys():
            val = x[index][term] / freq_sum * log(size / idf_res[term])
            if val > threshold:
                new_x[index][term] = val
    return new_x


if __name__ == '__main__':
    x = {0: {1250536: 1},
         1: {634175: 1,
             1095476: 4,
             805104: 1},
         2: {1250536: 1,
             805104: 1}}
    print(idf(x))
    print(tf_idf(x, 0.1))
