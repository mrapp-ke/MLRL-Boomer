#!/bin/sh

#SBATCH --job-name=enron_6
#SBATCH --output=python/tests/res/tmp/n-estimators_1/criterion_entropy/max-depth_5/dataset_enron/results/std.out
#SBATCH --error=python/tests/res/tmp/n-estimators_1/criterion_entropy/max-depth_5/dataset_enron/results/std.err
#SBATCH --time=2:30:00
#SBATCH --cpus-per-tasks=1

python -m venv .venv
. .venv/bin/activate
python -m pip install mlrl-testbed-sklearn
mlrl-testbed mlrl.testbed_sklearn --base-dir python/tests/res/tmp --criterion entropy --data-dir python/tests/res/data --dataset enron --estimator RandomForestClassifier --log-level debug --max-depth 5 --model-save-dir n-estimators_1/criterion_entropy/max-depth_5/dataset_enron/models --n-estimators 1 --parameter-save-dir n-estimators_1/criterion_entropy/max-depth_5/dataset_enron/parameters --result-dir n-estimators_1/criterion_entropy/max-depth_5/dataset_enron/results --save-evaluation true --save-meta-data false
deactivate
