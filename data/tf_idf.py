from math import log


def idf(original_x):
    idf_res = dict()
    for term_freq in original_x:
        for each_term in term_freq.keys():
            idf_res.setdefault(each_term, 0.0)
            idf_res[each_term] += 1
    return idf_res


def x_with_tf_idf(original_x, threshold):
    new_x = list()
    idf_res = idf(original_x)
    size = float(len(original_x))
    for each_instance in original_x:
        new_instance = dict()
        instance_freq_sum = float(sum(each_instance.values()))
        for term in each_instance.keys():
            val = each_instance[term] / instance_freq_sum * log(size / idf_res[term])
            if val >= threshold:
                new_instance[term] = val
        new_x.append(new_instance)
    return new_x
