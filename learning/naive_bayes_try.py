import numpy as np
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
x_col_index = [0, 3,
               0, 4,
               0, 4,
               0, 3,
               0, 3,
               1, 3,
               1, 4,
               1, 4,
               1, 5,
               1, 5,
               2, 5,
               2, 4,
               2, 4,
               2, 5,
               2, 4]
x_dat = [1] * 30
x_row_index = list()
for i in range(15):
    x_row_index.extend([i] * 2)
x = csr_matrix((x_dat, (x_row_index, x_col_index)), shape=(15, 6))
bnb = BernoulliNB()
bnb.fit(x, y)
print(bnb.predict(np.array([1, 0, 0, 1, 0, 0])))

mnb = MultinomialNB()
mnb.fit(x, y)
print(mnb.predict(x))
print(mnb.predict(np.array([1, 0, 0, 1, 0, 0])))
print(mnb.predict(np.array([0, 0, 1, 0, 1, 0])))
