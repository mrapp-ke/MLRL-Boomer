#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of arguments!"
    exit 1
fi

# Command line arguments
DATASET=$1

echo "./slurm.sh ${DATASET} laplace precision"
echo "./slurm.sh ${DATASET} f-measure{\'beta\':0.25} precision"
echo "./slurm.sh ${DATASET} laplace irep"
echo "./slurm.sh ${DATASET} f-measure{\'beta\':0.25} irep"
