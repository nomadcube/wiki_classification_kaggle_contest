from collections import defaultdict


def occurrence(y):
    res = defaultdict(set)
    for i, labels in enumerate(y):
        res[labels].add(i)
    return res
