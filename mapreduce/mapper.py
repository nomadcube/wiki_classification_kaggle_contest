#!/usr/bin/env python
import sys


vec = [7, 8, 9]
for line in sys.stdin:
    row_index, col_index, val = line.strip().split(",")
    row_index = int(row_index)
    col_index = int(col_index)
    val = float(val)
    res = val * vec[col_index]
    print('{0},{1}'.format(row_index, res))
