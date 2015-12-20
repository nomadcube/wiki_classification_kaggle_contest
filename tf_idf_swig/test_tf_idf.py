import tf_idf
import time

from data_processing.tf_idf import term_frequency


a = tf_idf.term_val_t({1: 1.0, 2: 2.0, 3: 3.0})
print(tf_idf.val_sum(a))

b = tf_idf.doc_term_val_t({0: a, 1: a})

start_time = time.time()
c = tf_idf.term_frequency(b)
print(time.time() - start_time)

d = {0: {1: 1.0, 2: 2.0, 3: 3.0}, 1: {1: 1.0, 2: 2.0, 3: 3.0}}
start_time = time.time()
e = term_frequency(d)
print(time.time() - start_time)
