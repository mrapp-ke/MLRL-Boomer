#!/bin/sh

#SBATCH --output=python/tests/res/tmp/instance-sampling_none/loss_squared-error-decomposable/dataset_atp7d-binary/results/std.out
#SBATCH --error=python/tests/res/tmp/instance-sampling_none/loss_squared-error-decomposable/dataset_atp7d-binary/results/std.err
#SBATCH --time=2:30:00
#SBATCH --cpus-per-tasks=1

python -m venv .venv
. .venv/bin/activate
python -m pip install mlrl-testbed-sklearn
mlrl-testbed mlrl.boosting --base-dir python/tests/res/tmp --data-dir python/tests/res/data --dataset atp7d-binary --instance-sampling none --log-level debug --loss squared-error-decomposable --model-save-dir instance-sampling_none/loss_squared-error-decomposable/dataset_atp7d-binary/models --parameter-save-dir instance-sampling_none/loss_squared-error-decomposable/dataset_atp7d-binary/parameters --problem-type regression --result-dir instance-sampling_none/loss_squared-error-decomposable/dataset_atp7d-binary/results --save-evaluation true --save-meta-data false
deactivate
