def x_feature_set(x):
    """Get all distinct features in x."""
    feature_set = set()
    for each_instance in x.values():
        for feature in each_instance.keys():
            feature_set.add(feature)
    return feature_set

