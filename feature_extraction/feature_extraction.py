def extraction(x, threshold=0.0, x_subset=None):
    """Extract feature with value larger than threshold or in x_subset."""
    updated_x = dict()
    for instance_index in x.keys():
        updated_x[instance_index] = dict()
        for feature, val in x[instance_index].items():
            if not x_subset:
                if val <= threshold:
                    continue
            else:
                if feature not in x_subset:
                    continue
            updated_x[instance_index][feature] = val
    return updated_x
