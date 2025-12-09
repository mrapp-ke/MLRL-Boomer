#!/bin/sh

#SBATCH --job-name=enron_5
#SBATCH --output=python/tests/res/tmp/n-estimators_1/max-leaf-nodes_10/dataset_enron/results/std.out
#SBATCH --error=python/tests/res/tmp/n-estimators_1/max-leaf-nodes_10/dataset_enron/results/std.err
#SBATCH --time=2:30:00
#SBATCH --cpus-per-tasks=1

python -m venv .venv
. .venv/bin/activate
python -m pip install mlrl-testbed-sklearn
mlrl-testbed mlrl.testbed_sklearn --base-dir python/tests/res/tmp --data-dir python/tests/res/data --dataset enron --estimator RandomForestRegressor --log-level debug --max-leaf-nodes 10 --model-save-dir n-estimators_1/max-leaf-nodes_10/dataset_enron/models --n-estimators 1 --parameter-save-dir n-estimators_1/max-leaf-nodes_10/dataset_enron/parameters --problem-type regression --result-dir n-estimators_1/max-leaf-nodes_10/dataset_enron/results --save-evaluation true --save-meta-data false
deactivate
