#!/bin/bash

if [[ "$#" -lt 3 || "$#" -gt 6 ]]; then
    echo "Illegal number of arguments!"
    exit 7
fi

# Command line arguments
ALGO=$1
DATASET=$2
HEAD_REFINEMENT_OR_TYPE=$3
HEURISTIC=$4
PRUNING_HEURISTIC=$5
REFINE_HEADS=$6

# Constants
MEMORY=2048
CORES=1
LOG_LEVEL="ERROR"
MAX_RULES=500
FOLDS=10
INSTANCE_SUB_SAMPLING="seco-random-instance-selection"
TIME_LIMIT_HOURS=24
TIME_LIMIT_SECONDS=$((TIME_LIMIT_HOURS*3600))

# Paths
ROOT_DIR="${PWD}"
DATA_DIR="${ROOT_DIR}/data"
SUB_DIR="${DATASET}_pruning_${HEAD_REFINEMENT_OR_TYPE::1}"

if [[ -n $HEURISTIC ]]; then  # when using seco append the heuristic
  SUB_DIR="${SUB_DIR}_${HEURISTIC::1}"
fi
if [[ -n $PRUNING_HEURISTIC ]]; then # when pruning append the prune heuristic
  SUB_DIR="${SUB_DIR}_${PRUNING_HEURISTIC::1}"
fi
if [[ -n $REFINE_HEADS ]]; then # when refining heads append rh
  SUB_DIR="${SUB_DIR}_rh"
fi

LOG_DIR="${ROOT_DIR}/results/${SUB_DIR}/logs"
OUTPUT_DIR="${ROOT_DIR}/results/${SUB_DIR}/evaluation"
MODEL_DIR="${ROOT_DIR}/models/${SUB_DIR}"
WORK_DIR="${LOG_DIR}"

# Parameters
PARAMETERS="--log-level ${LOG_LEVEL} --data-dir ${DATA_DIR} --dataset ${DATASET} --output-dir ${OUTPUT_DIR} --folds ${FOLDS} --current-fold \$SLURM_ARRAY_TASK_ID --max-rules ${MAX_RULES} --time-limit ${TIME_LIMIT_SECONDS}"

if [[ $ALGO == "main_jrip" ]]; then  # Parameters for main_jrip
  PARAMETERS="${PARAMETERS} --print-rules false --riper ${HEAD_REFINEMENT_OR_TYPE}"
else  # Parameters for main_seco
  PARAMETERS="${PARAMETERS} --model-dir ${MODEL_DIR} --instance-sub-sampling ${INSTANCE_SUB_SAMPLING} --heuristic ${HEURISTIC} --head-refinement ${HEAD_REFINEMENT_OR_TYPE}"
fi

# Parameters for pruning
if [[ -n $PRUNING_HEURISTIC ]]; then
  PARAMETERS="${PARAMETERS} --pruning irep --pruning-heuristic ${PRUNING_HEURISTIC}"
fi

# Parameters for peak lift function for multi-label rules
if [[ $HEAD_REFINEMENT_OR_TYPE == "partial" ]]; then
  CARDINALITY=$(./peak_label.sh "$DATASET")

  if [[ $CARDINALITY == "dataset not supported" ]]; then
    echo "dataset not supported"
    exit 22
  fi

  PARAMETERS="${PARAMETERS} --lift-function peak\\{\\'peak_label\\':${CARDINALITY},\\'max_lift\\':1.08,\\'curvature\\':1\\}"
fi

# Parameter for head refinement
if [[ -n $REFINE_HEADS ]]; then
  PARAMETERS="${PARAMETERS} --prune-head"
fi

# Create directories
echo "Creating directory ${LOG_DIR}"
mkdir -p "${LOG_DIR}"
echo "Creating directory ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"
echo "Creating directory ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Create slurm file
FILE="${SUB_DIR}.sh"
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
  echo "${ROOT_DIR}/venv/bin/python3 ${ROOT_DIR}/python/${ALGO}.py ${PARAMETERS}"
} >> "$FILE"

# Run SLURM jobs
sbatch "$FILE"
rm "$FILE"
echo "Started array of experiments with parameters ${PARAMETERS}"
