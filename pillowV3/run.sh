#!/bin/bash

# form main.py:
#batch_size = sys.argv[1]
#size_1 = sys.argv[2]
#size_2 = sys.argv[3]
#batch_training_size = sys.argv[4]

while
	batch_size=$((RANDOM % 90 + 10))
	size_1=$((RANDOM % 40 + 10))
	size_2=$((RANDOM % 40 + 10))
	batch_training_size=$((RANDOM % 50 + 5))
	./main.py $batch_size $size_1 $size_2 $batch_training_size
do
	sleep 1
done
