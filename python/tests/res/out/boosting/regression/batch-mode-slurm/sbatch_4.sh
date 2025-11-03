#!/bin/sh

#SBATCH --output=python/tests/res/tmp/instance-sampling_with-replacement{sample_size=0.5}/loss_squared-error-decomposable/feature-format_sparse/dataset_atp7d/results/std.out
#SBATCH --error=python/tests/res/tmp/instance-sampling_with-replacement{sample_size=0.5}/loss_squared-error-decomposable/feature-format_sparse/dataset_atp7d/results/std.err
#SBATCH --time=2:30:00
#SBATCH --cpus-per-tasks=1

python -m venv .venv
. .venv/bin/activate
python -m pip install mlrl-testbed-sklearn
mlrl-testbed mlrl.boosting --base-dir python/tests/res/tmp --data-dir python/tests/res/data --dataset atp7d --feature-format sparse --instance-sampling with-replacement\{sample_size=0.5\} --log-level debug --loss squared-error-decomposable --model-save-dir instance-sampling_with-replacement\{sample_size=0.5\}/loss_squared-error-decomposable/feature-format_sparse/dataset_atp7d/models --parameter-save-dir instance-sampling_with-replacement\{sample_size=0.5\}/loss_squared-error-decomposable/feature-format_sparse/dataset_atp7d/parameters --problem-type regression --result-dir instance-sampling_with-replacement\{sample_size=0.5\}/loss_squared-error-decomposable/feature-format_sparse/dataset_atp7d/results --save-evaluation true --save-meta-data false
deactivate
