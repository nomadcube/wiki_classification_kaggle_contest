from time import time

import hierarchy

start_time = time()
ht = hierarchy.HierarchyTable()
ht.read_data('/Users/wumengling/PycharmProjects/kaggle/unit_test_data/fake_hierarchy.txt')
ht.update(2)
print(time() - start_time)
print(ht.dat.keys())
