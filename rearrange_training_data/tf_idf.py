from math import log


def log_inverse_doc_frequency(train_data_path):
    """Calculate count of documents which contained each feat."""
    res = dict()
    doc_count = 0.0
    with open(train_data_path, 'r') as train_data:
        for line in train_data.readlines()[1:]:
            doc_count += 1
            for column in line.strip().split(' '):
                if ':' not in column:
                    continue
                term = column.split(':')[0]
                res.setdefault(term, 0.0)
                res[term] += 1
    for term in res.keys():
        res[term] = log(doc_count / res[term])
    return res


def term_frequency(train_data_path):
    """Generate tf-idf value for each feat in each doc."""
    res = dict()
    with open(train_data_path, 'r') as train_data:
        for row_index, line in enumerate(train_data.readlines()[1:]):
            res[row_index] = dict()
            for column in line.strip().split(' '):
                if ':' not in column:
                    continue
                term, count = column.split(':')
                res[row_index][term] = float(count)
            all_feat_sum = sum(res[row_index].values())
            for term in res[row_index].keys():
                res[row_index][term] /= all_feat_sum
    return res
