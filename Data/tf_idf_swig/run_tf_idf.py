import time

import tf_idf

a = tf_idf.term_val_t({1: 1.0, 2: 2.0, 3: 3.0})
print(tf_idf.val_sum(a))

b = tf_idf.doc_term_val_t({0: a, 1: a})

x_tf_idf = tf_idf.tf_idf(b, -1)
print(dict(x_tf_idf[0]))
print(dict(x_tf_idf[1]))
