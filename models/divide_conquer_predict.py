import heapq


class OneLabelScore(object):
    __slots__ = ('label', 'score')

    def __init__(self, label, score):
        self.label = label
        self.score = score

    def __lt__(self, other):
        return self.score > other.score


class OneSamplePrediction:
    def __init__(self):
        self._data = list()

    def push(self, one_label_score):
        self._data.append(one_label_score)

    def pop_max(self, k):
        heapq.heapify(self._data)
        self._data.sort()
        return [d.label for d in self._data[:k]]

    def __len__(self):
        return len(self._data)


class AllSamplePrediction:
    def __init__(self, num_sample):
        self._predict_list = [OneSamplePrediction() for _ in range(num_sample)]

    def push(self, sample_no, one_label_score):
        self._predict_list[sample_no].push(one_label_score)

    def pop_max(self, k):
        return [one_sample_predict.pop_max(k) for one_sample_predict in self._predict_list]

    def __len__(self):
        return len(self._predict_list)


if __name__ == '__main__':
    a_op = OneLabelScore('a', 1.0)
    b_op = OneLabelScore('b', 2.0)
    c_op = OneLabelScore('c', 3.0)

    h = OneSamplePrediction()
    h.push(a_op)
    h.push(b_op)
    h.push(c_op)

    print h.pop_max(2)

    all_sample_predict = AllSamplePrediction(2)
    all_sample_predict.push(0, a_op)
    all_sample_predict.push(0, b_op)
    all_sample_predict.push(0, c_op)
    all_sample_predict.push(1, OneLabelScore('d', 0.4))
    all_sample_predict.push(1, OneLabelScore('e', 5.0))
    all_sample_predict.push(1, OneLabelScore('f', 0.6))

    print len(all_sample_predict)
    print all_sample_predict.pop_max(2)
