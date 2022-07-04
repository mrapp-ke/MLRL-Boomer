"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython._validation import assert_greater_or_equal


cdef class SequentialPostOptimizationConfig:
    """
    Allows to configure a method that optimizes each rule in a model by relearning it in the context of the other rules.
    Multiple iterations, where the rules in a model are relearned in the order of their induction, may be carried out.
    """

    def get_num_iterations(self) -> int:
        """
        Returns the number of iterations that are performed for optimizing a model.

        :return: The number of iterations that are performed for optimizing a model
        """
        return self.config_ptr.getNumIterations()

    def set_num_iterations(self, num_iterations: int) -> SequentialPostOptimizationConfig:
        """
        Sets the number of iterations that should be performed for optimizing a model.

        :param num_iterations:  The number of iterations to be performed. Must be at least 1
        :return:                An `SequentialPostOptimizationConfig` that allows further configuration of the
                                optimization method
        """
        assert_greater_or_equal('num_iterations', num_iterations, 1)
        self.config_ptr.setNumIterations(num_iterations)
        return self
