from hierarchy_processing import part_hierarchy, p_hierarchy, correct_wrong_forest, update_inner_nodes


def test_read_file():
    lines = part_hierarchy('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/fake_hierarchy.txt')
    assert len(lines) == 9
    assert lines == ['1 2',
                     '1 3',
                     '2 4',
                     '2 5',
                     '5 6',
                     '5 7',
                     '5 8',
                     '9 10',
                     '9 11']


def test_root_node():
    h_dat = ['1 2',
             '1 3',
             '2 4',
             '2 5',
             '5 6',
             '5 7',
             '5 8',
             '9 10',
             '9 11']
    res = p_hierarchy(h_dat)
    assert len(res) == 2
    assert res == {'1', '9'}


def test_correct_wrong_forest():
    h_dat = ['1 2',
             '1 3',
             '2 4',
             '2 5',
             '5 6',
             '5 7',
             '5 8',
             '9 10',
             '9 11']
    roots = {'1', '9'}
    res = correct_wrong_forest(h_dat, roots)
    assert len(res) == 2
    assert res[0]['1'] == {'2', '3'}
    assert res[0]['9'] == {'10', '11'}
    assert res[1]['2'] == {'4', '5'}
    assert res[1]['5'] == {'6', '7', '8'}


def test_correct_tree_update():
    correct_t = {'2', '3'}
    wrong_f = {'5': {'6', '7', '8'},
               '2': {'4', '5'}}
    new_correct_tree = update_inner_nodes(correct_t, wrong_f)
    assert len(new_correct_tree) == 4
    assert new_correct_tree == {'2', '3', '4', '5'}
