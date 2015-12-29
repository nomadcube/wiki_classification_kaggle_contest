#!/usr/bin/env bash
input_data='/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv'
awk -F" " '{print $1}' ${input_data} | sort | uniq -c | sort -k 1 -r | head

#样本数  类别
#5692 24177
#3322 444502,87241
#3271 228232
#2640 73462
#2478 444502
#2422 383600
#2088 316670
#1821 174425
#1604 285613,24177
#1274 185999
#
#可以将原样本转换为"如果真实类型包含24177则为1，否则为0"。
