#!/usr/bin/env bash

# number of non zero element
cat /Users/wumengling/PycharmProjects/kaggle/input_data/train.csv | tr -d "\n" | tr ":" "\n"| wc -l
# 100611105

# number of row
wc -l /Users/wumengling/PycharmProjects/kaggle/input_data/train.csv
# 2365436

# number of column (using python)
# 2085166

# sparsity
# 100611105.0 / (2365436.0 * 2085166.0) = 2.04e-05

# test if 100611105 = nnz < (m*(n-1)-1)/2 = 2466162178469
# true
