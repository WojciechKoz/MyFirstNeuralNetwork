#!/bin/bash

# form main.py:
#batch_size = sys.argv[1]
#size_1 = sys.argv[2]
#size_2 = sys.argv[3]
#batch_training_size = sys.argv[4]

while
	batch_size=$((RANDOM % 80 + 50))
	size_1=$((RANDOM % 20 + 20))
	size_2=$((RANDOM % 15 + 20))
	batch_training_size=$((RANDOM % 300 + 250))
	./main.py $batch_size $size_1 $size_2 $batch_training_size
do
	sleep 1
done
