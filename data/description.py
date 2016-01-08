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
        res.setdefault(each_y, 0)
        res[each_y] += 1
    return res


def describe_x_y(x, y):
    description = namedtuple('description', 'sample_size feature_dimension class_distribution')
    if not isinstance(x, list):
        raise TypeError()
    if not isinstance(y, list):
        raise TypeError()
    sample_size = len(x)
    feature_dimension = len(collect_feature(x))
    class_distribution = each_class_count(y)
    return description(sample_size, feature_dimension, class_distribution)
