#!/bin/sh

#SBATCH --job-name=enron_6
#SBATCH --output=python/tests/res/tmp/instance-sampling_none/heuristic_m-estimate/feature-format_sparse/dataset_enron/results/std.out
#SBATCH --error=python/tests/res/tmp/instance-sampling_none/heuristic_m-estimate/feature-format_sparse/dataset_enron/results/std.err
#SBATCH --time=2:30:00
#SBATCH --cpus-per-tasks=1

python -m venv .venv
. .venv/bin/activate
python -m pip install mlrl-testbed-sklearn
mlrl-testbed mlrl.seco --base-dir python/tests/res/tmp --data-dir python/tests/res/data --dataset enron --feature-format sparse --heuristic m-estimate --instance-sampling none --log-level debug --model-save-dir instance-sampling_none/heuristic_m-estimate/feature-format_sparse/dataset_enron/models --parameter-save-dir instance-sampling_none/heuristic_m-estimate/feature-format_sparse/dataset_enron/parameters --result-dir instance-sampling_none/heuristic_m-estimate/feature-format_sparse/dataset_enron/results --save-evaluation true --save-meta-data false
deactivate
