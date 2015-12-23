from tf_idf_swig.tf_idf import tf_idf, doc_term_val_t
from random import randrange
import io


class Data:
    """Represent total data."""

    def __init__(self, data_path):
        self.y, self.x = self._get_y_x(data_path)
        self.y_remapping_rel = dict()
        self.train_y = list()
        self.train_x = list()
        self.test_y = list()
        self.test_x = list()

    def __len__(self):
        """Take the size of y or x to be Data's size."""
        if len(self.y) != len(self.x):
            raise ValueError('The size of y and x must agree.')
        return len(self.y)

    def _get_y_x(self, train_data_path):
        """Get y and x data from file lines."""
        y_res = dict()
        x_res = dict()
        for index, line in enumerate(io.open(train_data_path, 'r').readlines()):
            x_res[index] = dict()
            tmp_y, tmp_x = line.strip().split(' ', 1)
            y_res[index] = tmp_y
            for column in tmp_x.split(' '):
                feat, val = column.split(':')
                feat = int(feat)
                val = float(val)
                x_res[index][feat] = val
        return y_res, x_res

    def remap(self):
        """Map original y value to its first index, return mapping relation and update y."""
        for y_index, y_str in enumerate(self.y.values()):
            if y_str not in self.y_remapping_rel:
                self.y_remapping_rel[y_str] = y_index
        for y_key in self.y.keys():
            self.y[y_key] = self.y_remapping_rel[self.y[y_key]]
        return self

    def dim_reduction(self, threshold):
        """Use tf-idf to perform dimension reduction, update x."""
        self.x = dict(tf_idf(doc_term_val_t(self.x), threshold))
        for k in self.x:
            self.x[k] = dict(self.x[k])
        return self

    def sample_split(self, train_count):
        """Generate train and test sample from total data."""
        sample = self._sampling(train_count)
        self.train_y = [self.y[i] for i in sample]
        self.train_x = [self.x[i] for i in sample]
        self.test_y = [self.y[j] for j in self.y.keys() if j not in sample]
        self.test_x = [self.x[j] for j in self.x.keys() if j not in sample]

    def _sampling(self, size):
        """Take at most `size` sample."""
        sample_result = set()
        for t in range(size):
            sample_result.add(randrange(len(self)))
        return sample_result

    def _feature_count(self):
        """Count distinct features in the training data."""
        feature_set = set()
        for instance_index in self.x.keys():
            for feature in self.x[instance_index].keys():
                feature_set.add(feature)
        return len(feature_set)

    def description(self):
        """Describe total data."""
        return len(self), self._feature_count()
