"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""


cdef class SeCoRuleLearnerConfig:
    """
    Allows to configure a rule learner that makes use of the separate-and-conquer (SeCo) paradigm.
    """

    cdef ISeCoRuleLearnerConfig* get_seco_rule_learner_config_ptr(self):
        pass    
