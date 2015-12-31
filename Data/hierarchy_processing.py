from time import time


class LineObject:
    def __init__(self, parent_id, own_id):
        self.parent_id = parent_id
        self.own_id = own_id


class Node:
    def __init__(self, line_object):
        self.element = line_object
        self.parent = None


class HierarchyTable:
    def __init__(self):
        self.dat = dict()

    def read_data(self, f_path):
        with open(f_path, 'r') as f_stream:
            for line in f_stream.readlines():
                p_id, o_id = line.strip().split(' ')
                self.dat[o_id] = Node(LineObject(p_id, o_id))
        return self

    def update(self):
        for line_object_id, parent_node in self.dat.items():
            original_parent_id = parent_node.element.parent_id
            while original_parent_id in self.dat.keys():
                updated_parent_node = self.dat[original_parent_id]
                self.dat[line_object_id] = updated_parent_node
                original_parent_id = updated_parent_node.element.parent_id
        return self


if __name__ == '__main__':
    start_time = time()
    h_table = HierarchyTable()
    h_table.read_data('/Users/wumengling/PycharmProjects/kaggle/input_data/hierarchy_sample.txt')
    h_table.update()
    print(time() - start_time)
