#!/usr/bin/env python
import sys

# multiplication between matrix and vector
row_sum = dict()
for line in sys.stdin:
    row_index, val = line.strip().split(',')
    val = float(val)
    row_sum.setdefault(row_index, 0.0)
    row_sum[row_index] += val

for k, v in row_sum.items():
    print '%s\t%s' % (k, v)
