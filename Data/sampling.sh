#!/usr/bin/env bash

sample_path="/Users/wumengling/PycharmProjects/kaggle/input_data/train.csv"
sample_result_path="/Users/wumengling/PycharmProjects/kaggle/input_data/random_sampling_train.csv"
sample_count=`wc -l ${sample_path} | awk -F" " '{print $1}' | cat `
count=1
threshold=1000

echo
echo "$sample_count random numbers:"
echo "-----------------"
while [ "$count" -le ${sample_count} ]
do
  number=$RANDOM
  if [ "$number" -le ${threshold} ]
  then
    echo ${number}
    sed -n "$count p" ${sample_path} | cat >> ${sample_result_path}
  fi
  let "count += 1"
done
echo "-----------------"
