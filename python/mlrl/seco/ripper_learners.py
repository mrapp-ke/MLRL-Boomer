#!/usr/bin/python

import numpy as np
import wittgenstein as lw

from sklweka.dataset import to_nominal_labels
from skmultilearn.problem_transform import BinaryRelevance
from sklweka.classifiers import WekaEstimator

from mlrl.common.learners import Learner


class LwRipper(Learner):
    """
    Wrapper for wittgenstein's ripper to implement Learner and with it sklearn BaseEstimator necessary for the
    sklearn BinaryRelevance.
    """

    def __init__(self, max_rules, random_state):
        self.max_rules = max_rules
        self.random_state = random_state
        self.learner = None

    def _fit(self, x, y):
        self.learner = lw.RIPPER(max_rules=self.max_rules, random_state=self.random_state)
        return self.learner.fit(x, y)

    def _predict(self, x):
        return self.learner.predict(x)

    def get_name(self) -> str:
        return 'wittgenstein_RIPPER_max-rules=' + str(self.max_rules) + '_random_state=' + str(self.random_state)


class WekaRipper(Learner):

    def __init__(self, random_state):
        self.random_state = random_state
        self.learner = None

    def _fit(self, x, y):
        self.learner = WekaEstimator(classname="weka.classifiers.rules.JRip", options=["-S", str(self.random_state)])
        y = to_nominal_labels(y)
        rules = self.learner.fit(x, y)
        return rules

    def _predict(self, x):
        prediction = self.learner.predict(x)
        prediction = np.array([1 if "1" in y else 0 for y in prediction])
        return prediction

    def get_name(self) -> str:
        return 'WEKA_RIPPER_random_state=' + str(self.random_state)


def create_learner(ripper_type: str, random_state: int, max_rules: int):
    if ripper_type == 'weka':
        return WekaRipper(random_state)
    return LwRipper(max_rules, random_state)


class RipperRuleLearner(Learner):

    def __init__(self, max_rules: int = 500, random_state: int = 1, ripper: str = 'weka'):
        self.max_rules = max_rules
        self.random_state = random_state
        self.ripper = ripper
        self.learner = None

    def _fit(self, x, y):
        self.learner = BinaryRelevance(
            create_learner(self.ripper, self.random_state, self.max_rules),
            require_dense=[True, True]
        )
        return self.learner.fit(x, y)

    def _predict(self, x):
        return self.learner.predict(x)

    def get_name(self) -> str:
        return self.ripper + '_RIPPER_mlc_max-rules=' + str(self.max_rules) + '_random_state=' + str(self.random_state)
