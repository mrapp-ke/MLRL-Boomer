"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow running experiments.
"""
from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.experiment_sklearn import SkLearnExperiment
from mlrl.testbed.experiments.problem_domain import ClassificationProblem, ProblemDomain, RegressionProblem
from mlrl.testbed.experiments.problem_domain_sklearn import SkLearnClassificationProblem, SkLearnProblem, \
    SkLearnRegressionProblem
