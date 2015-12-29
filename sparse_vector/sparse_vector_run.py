from sparse_vector import sparse_vector_dat_t, SparseVector

sv_dat = sparse_vector_dat_t({0: 1.2, 1: 1.2})
sv = SparseVector(sv_dat)
sv1 = sv.dot_multiplication(0.1)
print(dict(sv1.dat))

sv2 = sv.dot_multiplication(sv)
print(dict(sv2.dat))

f = sv.inner_product(sv)
print(f)

rhs_vec = SparseVector(sparse_vector_dat_t({1: 1.0}))
plus_vec = sv + rhs_vec
print(dict(plus_vec.dat))
