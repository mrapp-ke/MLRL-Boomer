"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from common.cython.measures cimport EvaluationMeasure

from libcpp.memory cimport make_shared


cdef class StoppingCriterion:
    """
    A wrapper for the pure virtual C++ class `IStoppingCriterion`.
    """
    pass


cdef class SizeStoppingCriterion(StoppingCriterion):
    """
    A wrapper for the C++ class `SizeStoppingCriterion`.
    """

    def __cinit__(self, uint32 max_rules):
        """
        :param max_rules: The maximum number of rules
        """
        self.stopping_criterion_ptr = <shared_ptr[IStoppingCriterion]>make_shared[SizeStoppingCriterionImpl](max_rules)


cdef class TimeStoppingCriterion(StoppingCriterion):
    """
    A wrapper for the C++ class `TimeStoppingCriterion`.
    """

    def __cinit__(self, uint32 time_limit):
        """
        :param time_limit: The time limit in seconds
        """
        self.stopping_criterion_ptr = <shared_ptr[IStoppingCriterion]>make_shared[TimeStoppingCriterionImpl](time_limit)


cdef class MeasureStoppingCriterion(StoppingCriterion):
    """
    A wrapper for the C++ class `MeasureStoppingCriterion`.
    """

    def __cinit__(self, EvaluationMeasure measure, uint32 update_interval, uint32 stop_interval, uint32 buffer_size):
        """
        :param measure:         The measure that should be used to assess the quality of a model
        :param update_interval: The interval to be used to update the quality of the current model, e.g., a value of 5
                                means that the model quality is assessed every 5 rules
        :param stop_interval:   The interval to be used to decide whether the induction of rules should be stopped,
                                e.g., a value of 10 means that the rule induction might be stopped after 10, 20, ...
                                rules. Must be a multiple of `updateInterval`
        :param buffer_size:     The number of quality scores to be stored in a buffer. Must be at least 1
        """
        cdef shared_ptr[IEvaluationMeasure] measure_ptr = measure.get_evaluation_measure_ptr()
        self.stopping_criterion_ptr = <shared_ptr[IStoppingCriterion]>make_shared[MeasureStoppingCriterionImpl](
            measure_ptr, update_interval, stop_interval, buffer_size)
