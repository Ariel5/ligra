#!/bin/bash

# Array of the number of threads to use
threads=(2 4 8 16 24)

# Loop over the number of threads
for t in "${threads[@]}"
do
    echo "Running with $t threads"
    export OMP_NUM_THREADS=$t
    bash run_gee.sh
done
