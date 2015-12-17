from math import log


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


def dim_reduction_with_tf_idf(x, term_set, threshold):
    """Yield each instance in dimension reduction training data using tf-idf."""
    tf = term_frequency(x)
    idf = log_inverse_doc_frequency(term_set, x)
    dim_reduction_x = dict()
    for instance_index in x.keys():
        dim_reduction_x[instance_index] = dict()
        for term in tf[instance_index].keys():
            updated_val = idf[term] * tf[instance_index][term]
            if updated_val <= threshold:
                continue
            dim_reduction_x[instance_index][term] = updated_val
    return dim_reduction_x
