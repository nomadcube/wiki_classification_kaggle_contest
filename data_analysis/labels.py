from collections import namedtuple, defaultdict


def occurrence(y):
    res = defaultdict(set)
    for i, labels in enumerate(y):
        for j, each_label in enumerate(labels):
            res[each_label].add(i)
    return res


def distribution(observations):
    occurrence_description = namedtuple("description", "cnt_total cnt_one max min")
    cnt = {k: len(v) for k, v in observations.items()}
    return occurrence_description(len(cnt), cnt.values().count(1.), max(cnt.values()), min(cnt.values()))


def most_frequent_label(y, top_k):
    occurrence_cnt = {k: len(v) for k, v in occurrence(y).items()}
    return sorted(occurrence_cnt, key=lambda x: occurrence_cnt[x], reverse=True)[:top_k]


if __name__ == '__main__':
    y = [[314523, 165538, 416827], [21631], [76255, 165538]]
    oy = occurrence(y)
    print oy
    print distribution(oy)
    for label, instances in oy.items():
        if len(instances) > 1:
            print instances.pop()
    print oy
