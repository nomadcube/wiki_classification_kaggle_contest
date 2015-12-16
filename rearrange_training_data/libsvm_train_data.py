def training_label(train_data_path):
    """Yield true label in training data."""
    with open(train_data_path, 'r') as train_data:
        for line in train_data.readlines()[1:]:
            label_list = list()
            for column in line.strip().split(' '):
                if ':' in column:
                    continue
                label_list.append(int(column.strip(',')))
            yield sum(label_list)   # todo: using sum instead of string contact affect unit test.


def dimension_reduction_instance(tf, idf, threshold):
    """Yield each instance in dimension reduction training data using tf-idf."""
    for row_index in range(len(tf.keys())):
        term_tfidf = dict()
        for term in tf[row_index].keys():
            tf_idf_val = idf[term] * tf[row_index][term]
            if tf_idf_val <= threshold:
                continue
            term = int(term)
            term_tfidf[term] = tf_idf_val
        yield term_tfidf
