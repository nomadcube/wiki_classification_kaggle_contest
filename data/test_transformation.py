import transformation


def test_base_sample_reader():
    sample_f_path = '/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt'
    prop_enum = [1.0, 0.0]
    read_res = transformation.base_sample_reader(sample_f_path, prop_enum[0])
    assert len(read_res) == 2
    assert len(read_res.y) == 3
    assert len(read_res.x) == 3
    assert read_res.y == ['314523,165538,416827',
                          '21631',
                          '76255,335416']
    assert read_res.x == [{'1250536': 1},
                          {'634175': 1, '1095476': 4, '805104': 1},
                          {'1250536': 1, '805104': 1}]
    read_res = transformation.base_sample_reader(sample_f_path, prop_enum[1])
    assert len(read_res) == 2
    assert len(read_res.y) == 0
    assert len(read_res.x) == 0


def test_construct_csr_sample():
    x = [{'1250536': 1},
         {'634175': 1, '1095476': 4, '805104': 1},
         {'1250536': 1, '805104': 1}]
    csr_1 = transformation.construct_csr_sample(x)[0]
    assert csr_1.shape == (3, 4)


def test_flatting_multi_label():
    x = [{'1250536': 1},
         {'634175': 1, '1095476': 4, '805104': 1},
         {'1250536': 1, '805104': 1}]
    y = ['314523,165538,416827',
         '21631',
         '76255,335416']
    flat_y, flat_x, flat_relation = transformation.flatting_multi_label(y, x)
    assert len(flat_x) == 6
    assert len(flat_y) == 6
    assert len(flat_relation) == 3
    assert flat_relation == {0: {0, 1, 2},
                             1: {3},
                             2: {4, 5}}
    assert flat_x == [{'1250536': 1},
                      {'1250536': 1},
                      {'1250536': 1},
                      {'634175': 1, '1095476': 4, '805104': 1},
                      {'1250536': 1, '805104': 1},
                      {'1250536': 1, '805104': 1}]
    assert flat_y == ['314523',
                      '165538',
                      '416827',
                      '21631',
                      '76255',
                      '335416']


def test_assemble_y():
    flat_y = ['314523',
              '165538',
              '416827',
              '21631',
              '76255',
              '335416']
    flat_relation = {0: {0, 1, 2},
                     1: {3},
                     2: {4, 5}}
    new_y = transformation.assemble_y(flat_y, flat_relation)
    assert len(new_y) == 3
    assert new_y == ['314523,165538,416827',
                     '21631',
                     '76255,335416']
