import transformation
import read_into_csr


def test_base_sample_reader():
    sample_f_path = '/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt'
    prop_enum = [1.0, 0.0]
    read_res = transformation.sample_reader(sample_f_path, prop_enum[0])
    assert len(read_res) == 2
    assert len(read_res.y) == 3
    assert len(read_res.x) == 3
    assert read_res.y == [['314523', '165538', '416827'],
                          ['21631'],
                          ['76255', '335416']]
    assert read_res.x == [{'1250536': 1},
                          {'634175': 1, '1095476': 4, '805104': 1},
                          {'1250536': 1, '805104': 1}]
    read_res = transformation.sample_reader(sample_f_path, prop_enum[1])
    assert len(read_res) == 2
    assert len(read_res.y) == 0
    assert len(read_res.x) == 0


def test_construct_csr_sample():
    x = [{'1250536': 1},
         {'634175': 1, '1095476': 4, '805104': 1},
         {'1250536': 1, '805104': 1}]
    csr_1 = transformation.construct_csr_sample(x)[0]
    assert csr_1.shape == (3, 4)


def test_read_into_csr():
    sample_f_path = '/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt'
    csr_res = read_into_csr.read_into_csr(sample_f_path)
