import tf_idf

a = tf_idf.term_val_t({1: 1.0, 2: 2.0, 3: 3.0})
print(tf_idf.val_sum(a))
