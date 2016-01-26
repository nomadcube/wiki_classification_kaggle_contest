import reader


def test_read_sample():
    smp_1, smp_2 = reader.read_sample('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt', 3, 1)
    assert smp_1.y == [[314523, 165538, 416827]]
    assert smp_1.element_x == [1.0]
    assert smp_1.col_index_x == [1250536]
    assert smp_1.row_index_x == [0]
    assert smp_1.max_feature == 1250536
    assert smp_1.max_class_label == 416827
    assert smp_2.y == [[21631], [76255, 335416]]
    assert smp_2.element_x == [1.0, 4.0, 1.0, 1.0, 1.0]
    assert smp_2.col_index_x == [634175, 1095476, 805104, 1250536, 805104]
    assert smp_2.row_index_x == [0, 0, 0, 1, 1]
    assert smp_2.max_feature == 1250536
    assert smp_2.max_class_label == 335416
