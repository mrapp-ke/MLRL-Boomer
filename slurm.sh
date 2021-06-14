#!/bin/bash

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
    echo "Illegal number of arguments!"
    exit 1
fi

# Command line arguments
DATASET=$1
HEAD_REFINEMENT=$2
HEURISTIC=$3
PRUNING_HEURISTIC=$4

# Constants
MEMORY=2048
CORES=1
LOG_LEVEL="ERROR"
MAX_RULES=500
FOLDS=10
INSTANCE_SUB_SAMPLING="seco-random-instance-selection"
HEAD_REFINEMENT="single-label"
TIME_LIMIT_HOURS=24
TIME_LIMIT_SECONDS=$((TIME_LIMIT_HOURS*3600))

# Paths
ROOT_DIR="${PWD}"
DATA_DIR="${ROOT_DIR}/data"

SUB_DIR="${DATASET}_pruning_${HEURISTIC::1}_${PRUNING_HEURISTIC::1}"
LOG_DIR="${ROOT_DIR}/results/${SUB_DIR}/logs"
OUTPUT_DIR="${ROOT_DIR}/results/${SUB_DIR}/evaluation"
MODEL_DIR="${ROOT_DIR}/models/${SUB_DIR}"
WORK_DIR="${LOG_DIR}"

# Create directories
echo "Creating directory ${LOG_DIR}"
mkdir -p "${LOG_DIR}"
echo "Creating directory ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"
echo "Creating directory ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

FILE="${SUB_DIR}.sh"

PARAMETERS="--log-level ${LOG_LEVEL} --data-dir ${DATA_DIR} --dataset ${DATASET} --model-dir ${MODEL_DIR} --output-dir ${OUTPUT_DIR} --folds ${FOLDS} --current-fold \$SLURM_ARRAY_TASK_ID --max-rules ${MAX_RULES} --time-limit ${TIME_LIMIT_SECONDS} --instance-sub-sampling ${INSTANCE_SUB_SAMPLING} --heuristic ${HEURISTIC} --head-refinement ${HEAD_REFINEMENT}"

if [[ -n $PRUNING_HEURISTIC ]]; then
  PARAMETERS="${PARAMETERS} --pruning irep --pruning-heuristic ${PRUNING_HEURISTIC}"
fi

{
  echo "#!/bin/bash"
  echo "#SBATCH -a 1-${FOLDS}"
  echo "#SBATCH -J ${SUB_DIR}_fold=%a"
  echo "#SBATCH -D ${WORK_DIR}"
  echo "#SBATCH -t ${TIME_LIMIT_HOURS}:00:00"
  echo "#SBATCH -n 1"
  echo "#SBATCH -c ${CORES}"
  echo "#SBATCH -o fold_%a.log"
  echo "#SBATCH -e fold_%a.err"
  echo "#SBATCH --mem-per-cpu=${MEMORY}"
  echo "${ROOT_DIR}/venv/bin/python3 ${ROOT_DIR}/python/main_seco.py ${PARAMETERS}"
} >> "$FILE"

# Run SLURM jobs
sbatch "$FILE"
rm "$FILE"
echo "Started array of experiments with parameters ${PARAMETERS}"
