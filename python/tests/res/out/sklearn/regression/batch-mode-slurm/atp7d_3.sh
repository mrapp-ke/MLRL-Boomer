#!/bin/sh

#SBATCH --job-name=atp7d_3
#SBATCH --output=python/tests/res/tmp/n-estimators_100/max-depth_2/dataset_atp7d/results/std.out
#SBATCH --error=python/tests/res/tmp/n-estimators_100/max-depth_2/dataset_atp7d/results/std.err
#SBATCH --time=2:30:00
#SBATCH --cpus-per-tasks=1

python -m venv .venv
. .venv/bin/activate
python -m pip install mlrl-testbed-sklearn
mlrl-testbed mlrl.testbed_sklearn --base-dir python/tests/res/tmp --data-dir python/tests/res/data --dataset atp7d --estimator RandomForestRegressor --log-level debug --log-plain --log-width 120 --max-depth 2 --model-save-dir n-estimators_100/max-depth_2/dataset_atp7d/models --n-estimators 100 --parameter-save-dir n-estimators_100/max-depth_2/dataset_atp7d/parameters --problem-type regression --result-dir n-estimators_100/max-depth_2/dataset_atp7d/results --save-evaluation true --save-meta-data false
deactivate
