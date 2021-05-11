#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of arguments!"
    exit 1
fi

# Command line arguments
DATASET=$1

./slurm.sh "${DATASET}" "laplace" "precision"
./slurm.sh "${DATASET}" "f-measure{\\'beta\\':0.25}" "precision"
./slurm.sh "${DATASET}" "laplace" "irep"
./slurm.sh "${DATASET}" "f-measure{\\'beta\\':0.25}" "irep"
