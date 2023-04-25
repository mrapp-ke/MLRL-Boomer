"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""


cdef class BoostingRuleLearnerConfig:
    """
    Allows to configure a rule learner that makes use of gradient boosting.
    """

    cdef IBoostingRuleLearnerConfig* get_boosting_rule_learner_config_ptr(self):
        pass
