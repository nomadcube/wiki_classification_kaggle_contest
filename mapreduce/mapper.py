#!/usr/bin/env python
import sys


line_index = 0
for line in sys.stdin:
    line = line.strip()
    words = line.split(' ')
    for word in words:
        print '%s\t%s' % (str(line_index), str(float(word) * 0.5))
    line_index += 1
