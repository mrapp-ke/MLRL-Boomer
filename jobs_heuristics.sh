#!/bin/bash

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
    echo "Illegal number of arguments!"
    exit 1
fi

# Command line arguments
DATASET=$1
HEAD_REFINEMENT=${2:"single-label"}

./slurm.sh "${DATASET}" "${HEAD_REFINEMENT}" "laplace" "precision"
./slurm.sh "${DATASET}" "${HEAD_REFINEMENT}" "f-measure{\\'beta\\':0.25}" "precision"
./slurm.sh "${DATASET}" "${HEAD_REFINEMENT}" "laplace" "irep"
./slurm.sh "${DATASET}" "${HEAD_REFINEMENT}" "f-measure{\\'beta\\':0.25}" "irep"
./slurm.sh "${DATASET}" "${HEAD_REFINEMENT}" "laplace"
./slurm.sh "${DATASET}" "${HEAD_REFINEMENT}" "f-measure{\\'beta\\':0.25}"
./slurm.sh "${DATASET}" "${HEAD_REFINEMENT}" "precision"
