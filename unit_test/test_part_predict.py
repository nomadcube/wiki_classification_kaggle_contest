import heapq

from test_config import TestBase
from models.divide_conquer_predict import OneLabelScore, OneSamplePrediction, AllSamplePrediction


class TestMNB(TestBase):
    def test_OneLabelScore(self):
        a_op = OneLabelScore('a', 1.0)
        b_op = OneLabelScore('b', 2.0)
        c_op = OneLabelScore('c', 2.0)
        assert not a_op < b_op
        assert a_op > b_op
        assert not b_op < c_op

    def test_OneSamplePrediction(self):
        a_op = OneLabelScore('a', 1.0)
        b_op = OneLabelScore('b', 2.0)
        c_op = OneLabelScore('c', 3.0)
        h = OneSamplePrediction()
        h.push(a_op)
        h.push(b_op)
        h.push(c_op)
        assert len(h) == 3
        max_abc = h.pop_max(1)
        assert max_abc[0] == 'c'

    def test_AllSamplePrediction(self):
        all_sample_predict = AllSamplePrediction(2)
        all_sample_predict.push(0, OneLabelScore('a', 1.0))
        all_sample_predict.push(0, OneLabelScore('b', 2.0))
        all_sample_predict.push(0, OneLabelScore('c', 3.0))
        all_sample_predict.push(1, OneLabelScore('d', 0.4))
        all_sample_predict.push(1, OneLabelScore('e', 5.0))
        all_sample_predict.push(1, OneLabelScore('f', 0.6))
        assert len(all_sample_predict) == 2
        assert all_sample_predict.pop_max(2) == [['c', 'b'], ['e', 'f']]
