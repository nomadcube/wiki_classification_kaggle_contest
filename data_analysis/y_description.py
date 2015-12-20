from collections import namedtuple


def y_distribution(y_map):
    """Estimate the distribution of y."""
    distribution = dict()
    for y in y_map.values():
        distribution.setdefault(y, 0.0)
        distribution[y] += 1
    return distribution


def descriptive_analysis(dat_list):
    """Generate basic statistics for the given data."""
    basic_statistics = namedtuple('basic_statistics', 'min_val max_val median mean_val')
    dat_size = len(dat_list)
    sorted_dat = sorted(dat_list)
    min_val = sorted_dat[0]
    max_val = sorted_dat[dat_size - 1]
    median = sorted_dat[int(dat_size / 2.0) - 1]
    mean_val = sum(dat_list) / float(dat_size)
    return basic_statistics(min_val, max_val, median, mean_val)


if __name__ == '__main__':
    from data_processing.TrainData import TrainData
    import time
    y_group = list()
    start_time = time.time()
    tr = TrainData('/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv')
    print(time.time() - start_time)
    print(len(tr.y))
    s_time = time.time()
    y_dist = y_distribution(tr.y)
    print(time.time() - s_time)
    print(len(y_dist))
    for instance_set in y_dist:
        y_group.append(len(instance_set))
    print(descriptive_analysis(y_group))
