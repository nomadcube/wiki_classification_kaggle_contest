import data_preparation


def test_sample_reader():
    sample_f_path = '/Users/wumengling/PycharmProjects/kaggle/unit_test_data/sample.txt'
    prop_enum = [1.0, 0.0]
    read_res = data_preparation.sample_reader(sample_f_path, prop_enum[0])
    assert len(read_res) == 3
    read_res = data_preparation.sample_reader(sample_f_path, prop_enum[1])
    assert len(read_res) == 0


def test_construct_csr():
    row_index = range(2)
    col_and_val = [{0: 0.0, 1: 1.0}, {0: 2.0, 1: 3.0}]
    constraint_features = [None, {0}]
    csr_1 = data_preparation.construct_csr(row_index, col_and_val, constraint_features[0])
    assert csr_1.shape == (2, 2)
    csr_1 = data_preparation.construct_csr(row_index, col_and_val, constraint_features[1])
    assert csr_1.shape == (2, 1)
