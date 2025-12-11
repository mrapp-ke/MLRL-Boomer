#!/bin/sh

#SBATCH --job-name=emotions_1
#SBATCH --output=python/tests/res/tmp/instance-sampling_none/heuristic_f-measure/dataset_emotions/results/std.out
#SBATCH --error=python/tests/res/tmp/instance-sampling_none/heuristic_f-measure/dataset_emotions/results/std.err
#SBATCH --time=2:30:00
#SBATCH --cpus-per-tasks=1

python -m venv .venv
. .venv/bin/activate
python -m pip install mlrl-testbed-sklearn
mlrl-testbed mlrl.seco --base-dir python/tests/res/tmp --data-dir python/tests/res/data --dataset emotions --heuristic f-measure --instance-sampling none --log-level debug --model-save-dir instance-sampling_none/heuristic_f-measure/dataset_emotions/models --parameter-save-dir instance-sampling_none/heuristic_f-measure/dataset_emotions/parameters --result-dir instance-sampling_none/heuristic_f-measure/dataset_emotions/results --save-evaluation true --save-meta-data false
deactivate
