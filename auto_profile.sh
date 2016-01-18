#!/usr/bin/env bash


python -m cProfile -s 'tottime' /Users/wumengling/PycharmProjects/kaggle/learning/multinomial_naive_bayes.py | cat > teee.txt
#python -m memory_profiler /Users/wumengling/PycharmProjects/kaggle/learning/multinomial_naive_bayes.py | cat >> teee.txt
