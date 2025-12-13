#!/bin/sh

#SBATCH --job-name=emotions_1
#SBATCH --output=python/tests/res/tmp/n-estimators_1/criterion_gini/dataset_emotions/results/std_fold-%a.out
#SBATCH --error=python/tests/res/tmp/n-estimators_1/criterion_gini/dataset_emotions/results/std_fold-%a.err
#SBATCH --array=1-2
#SBATCH --time=2:30:00
#SBATCH --cpus-per-tasks=1

python -m venv .venv
. .venv/bin/activate
python -m pip install mlrl-testbed-sklearn
mlrl-testbed mlrl.testbed_sklearn --base-dir python/tests/res/tmp --criterion gini --data-dir python/tests/res/data --data-split cross-validation\{first_fold=${SLURM_ARRAY_TASK_ID},last_fold=${SLURM_ARRAY_TASK_ID},num_folds=2\} --dataset emotions --estimator RandomForestClassifier --log-level debug --model-save-dir n-estimators_1/criterion_gini/dataset_emotions/models --n-estimators 1 --parameter-save-dir n-estimators_1/criterion_gini/dataset_emotions/parameters --result-dir n-estimators_1/criterion_gini/dataset_emotions/results --save-evaluation true --save-meta-data false
deactivate
