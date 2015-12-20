from math import log
from data_analysis.x_description import x_feature_set


def term_frequency(x):
    """Generate tf-idf value for each feat in each doc."""
    res = dict()
    for instance_index in x.keys():
        res[instance_index] = dict()
        feat_val_sum = sum(x[instance_index].values())
        for term in x[instance_index].keys():
            res[instance_index][term] = x[instance_index][term] / feat_val_sum
    return res


def log_inverse_doc_frequency(term_set, x):
    """Calculate count of documents which contained each feat."""
    res = dict()
    for term in term_set:
        res.setdefault(term, 0.0)
        for instance_index in x.keys():
            if term in x[instance_index].keys():
                res[term] += 1
        res[term] = log(len(x) / res[term])
    return res


def tf_idf(x):
    """Calculate final tf-idf value."""
    tf_idf_x = dict()
    tf = term_frequency(x)
    term_set = x_feature_set(x)
    idf = log_inverse_doc_frequency(term_set, x)
    for instance_index in x.keys():
        tf_idf_x[instance_index] = dict()
        for term in tf[instance_index].keys():
            tf_idf_x[instance_index][term] = idf[term] * tf[instance_index][term]
    return tf_idf_x
