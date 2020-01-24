#!/bin/sh

# Command line arguments
DATASET=$1
LOSS=$2
HEAD_REFINEMENT=$3

# Constants
FOLDS=10
INSTANCE_SUB_SAMPLING="bagging"
FEATURE_SUB_SAMPLING="random-feature-selection"
NUM_RULES=5000
SHRINKAGE=0.1

# Paths
ROOT_DIR="${PWD}"
WORK_DIR="${ROOT_DIR}/results/${DATASET}/${LOSS}/${HEAD_REFINEMENT}"
LOG_DIR="${WORK_DIR}/logs"
MODEL_DIR="${ROOT_DIR}/models/${DATASET}"
DATA_DIR="${ROOT_DIR}/data"

MEMORY=4096

JOB_NAME="${DATASET}_${LOSS}_${HEAD_REFINEMENT}"
FILE="${JOB_NAME}.sh"
PARAMETERS="--data-dir ${DATA_DIR} --dataset ${DATASET} --output-dir ${WORK_DIR} --model-dir ${MODEL_DIR} --folds ${FOLDS} --instance-sub-sampling ${INSTANCE_SUB_SAMPLING} --feature-sub-sampling ${FEATURE_SUB_SAMPLING} --num-rules ${NUM_RULES} --shrinkage ${SHRINKAGE} --loss ${LOSS} --head-refinement ${HEAD_REFINEMENT}"

echo "$FILE"
echo "#!/bin/sh" >> "$FILE"
echo "#SBATCH -J ${JOB_NAME}" >> "$FILE"
echo "#SBATCH -D ${WORK_DIR}" >> "$FILE"
echo "#SBATCH -t 24:00:00" >> "$FILE"
echo "#SBATCH -c 1" >> "$FILE"
echo "#SBATCH -o /dev/null" >> "$FILE"
echo "#SBATCH -e logs/%J.log" >> "$FILE"
echo "#SBATCH --mem-per-cpu=${MEMORY}" >> "$FILE"
echo "#SBATCH --ntasks=1" >> "$FILE"
echo "${ROOT_DIR}/venv/bin/python3.7 ${ROOT_DIR}/python/main.py ${PARAMETERS}" >> "$FILE"
chmod +x "$FILE"

# Create directories
echo "Creating directory ${LOG_DIR}"
mkdir -p "${LOG_DIR}"
echo "Creating directory ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"

# Run SLURM job
sbatch "$FILE"
rm "$FILE"
echo "Started experiment with parameters ${PARAMETERS}"