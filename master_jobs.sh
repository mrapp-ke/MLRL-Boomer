#!/bin/bash

if [[ "$#" -ne 1 ]]; then
    echo "Illegal number of arguments!"
    exit 7
fi

# Command line arguments
DATASET=$1

./job_heuristics.sh "${DATASET}"
./job_heuristics.sh "${DATASET}" "partial"
./slurm "main_jrip" "${DATASET}" "weka"
./slurm "main_jrip" "${DATASET}" "wittgenstein"
