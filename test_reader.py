from array import array

import reader


def test_read_sample():
    smp_1, smp_2, smp_3 = reader.read_sample('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt', 1, 1)
    assert smp_1.y == [array('I', [314523L, 165538L, 416827L])]
    assert smp_1.element_x == array('f', [1.0])
    assert smp_1.col_index_x == array('I', [1250536L])
    assert smp_1.row_indptr_x == array('I', [0L, 1L])
    assert smp_1.max_feature == 1250536
    assert smp_1.max_class_label == 416827

    assert smp_2.y == [array('I', [21631L]), array('I', [76255L, 335416L])]
    assert smp_2.element_x == array('f', [1.0, 4.0, 1.0, 1.0, 1.0])
    assert smp_2.col_index_x == array('I', [634175L, 1095476L, 805104L, 1250536L, 805104L])
    assert smp_2.row_indptr_x == array('I', [0L, 3L, 5L])
    assert smp_2.max_feature == 1250536
    assert smp_2.max_class_label == 335416

    assert smp_3.y == [array('I', [76255L, 335416L])]
    assert smp_3.element_x == array('f', [1.0, 1.0])
    assert smp_3.col_index_x == array('I', [1250536L, 805104L])
    assert smp_3.row_indptr_x == array('I', [0L, 2L])
    assert smp_3.max_feature == 1250536
    assert smp_3.max_class_label == 335416


def test_convert_to_csr():
    smp_1 = reader.read_sample('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt', 1, 1)[0]
    csr_smp_1 = smp_1.convert_to_csr(1250536 + 1)
    assert csr_smp_1.shape == (1, 1250537)
    assert csr_smp_1.nnz == 1
    assert csr_smp_1[0, 1250536] == 1.0
