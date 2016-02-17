from collections import namedtuple


def occurrence(y):
    res = dict()
    for i, labels in enumerate(y):
        for each_label in labels:
            res.setdefault(each_label, set())
            res[each_label].add(i)
    return res


def distribution(observations):
    occurrence_description = namedtuple("description", "cnt_total cnt_one max min")
    cnt = {k: len(v) for k, v in observations.items()}
    return occurrence_description(len(cnt), cnt.values().count(1.), max(cnt.values()), min(cnt.values()))


if __name__ == '__main__':
    y = [[314523, 165538, 416827], [21631], [76255, 165538]]
    print occurrence(y)
    print distribution(occurrence(y))
