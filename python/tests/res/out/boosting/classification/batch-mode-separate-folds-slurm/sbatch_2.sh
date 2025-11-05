#!/bin/sh

#SBATCH --output=python/tests/res/tmp/instance-sampling_none/loss_logistic-decomposable/feature-format_sparse/dataset_emotions/results/std_fold-%a.out
#SBATCH --error=python/tests/res/tmp/instance-sampling_none/loss_logistic-decomposable/feature-format_sparse/dataset_emotions/results/std_fold-%a.err
#SBATCH --array=1-2
#SBATCH --time=2:30:00
#SBATCH --cpus-per-tasks=1

python -m venv .venv
. .venv/bin/activate
python -m pip install mlrl-testbed-sklearn
mlrl-testbed mlrl.boosting --base-dir python/tests/res/tmp --data-dir python/tests/res/data --data-split cross-validation\{first_fold=${SLURM_ARRAY_TASK_ID},last_fold=${SLURM_ARRAY_TASK_ID},num_folds=2\} --dataset emotions --feature-format sparse --instance-sampling none --log-level debug --loss logistic-decomposable --model-save-dir instance-sampling_none/loss_logistic-decomposable/feature-format_sparse/dataset_emotions/models --parameter-save-dir instance-sampling_none/loss_logistic-decomposable/feature-format_sparse/dataset_emotions/parameters --result-dir instance-sampling_none/loss_logistic-decomposable/feature-format_sparse/dataset_emotions/results --save-evaluation true --save-meta-data false
deactivate
