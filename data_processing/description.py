from collections import namedtuple


def collect_feature(x):
    features = set()
    for each_x in x:
        for each_feature in each_x.keys():
            features.add(each_feature)
    return features


def each_class_count(y):
    res = dict()
    for each_y in y:
        for label in each_y:
            res.setdefault(label, 0)
            res[label] += 1
    return res


def features(x):
    res = set()
    for each_x in x:
        for each_feature in each_x.keys():
            res.add(each_feature)
    return res


def describe_x_y(x, y):
    description = namedtuple('description', 'sample_size feature_dimension class_distribution all_feature')
    if not isinstance(x, list):
        raise TypeError()
    if not isinstance(y, list):
        raise TypeError()
    sample_size = len(x)
    feature_dimension = len(collect_feature(x))
    class_distribution = each_class_count(y)
    all_feature = features(x)
    return description(sample_size, feature_dimension, class_distribution, all_feature)
