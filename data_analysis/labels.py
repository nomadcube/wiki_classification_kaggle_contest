from collections import namedtuple


def occurrence_distribution(y):
    occurrence_description = namedtuple("description", "cnt_total cnt_one max min")
    occurrence = dict()
    for labels in y:
        for each_label in labels:
            occurrence.setdefault(each_label, 0)
            occurrence[each_label] += 1.
    return occurrence_description(len(occurrence), occurrence.values().count(1.), max(occurrence.values()),
                                  min(occurrence.values()))


if __name__ == '__main__':
    y = [[314523, 165538, 416827], [21631], [76255, 165538]]
    print occurrence_distribution(y)
