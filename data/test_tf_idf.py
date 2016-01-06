from data import tf_idf


def test_tf_idf():
    x = [{1250536: 1},
         {634175: 1,
          1095476: 4,
          805104: 1},
         {1250536: 1,
          805104: 1}]
    x_0 = tf_idf.x_with_tf_idf(x, 0)
    assert len(x_0) == 3
    assert x_0 == [{1250536: 0.4054651081081644},
                   {805104: 0.06757751801802739, 1095476: 0.7324081924454064, 634175: 0.1831020481113516},
                   {1250536: 0.2027325540540822, 805104: 0.2027325540540822}]
    x_1 = tf_idf.x_with_tf_idf(x, 0.3)
    assert len(x_1) == 3
    assert x_1 == [{1250536: 0.4054651081081644},
                   {1095476: 0.7324081924454064},
                   {}]
