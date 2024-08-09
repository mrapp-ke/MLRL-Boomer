"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython.validation import assert_greater, assert_greater_or_equal, assert_less


cdef class FixedPartialHeadConfig:
    """
    Allows to configure partial rule heads that predict for a predefined number of outputs.
    """

    def get_output_ratio(self) -> float:
        """
        Returns the percentage that specifies for how many outputs the rule heads predict.

        :return: The percentage that specifies for how many outputs the rule heads predict or 0, if the percentage is
                 calculated based on the average label cardinality
        """
        return self.config_ptr.getOutputRatio()

    def set_output_ratio(self, output_ratio: float) -> FixedPartialHeadConfig:
        """
        Sets the percentage that specifies for how many outputs the rule heads should predict.

        :param output_ratio:    A percentage that specifies for how many outputs the rule heads should predict, e.g., if
                                100 outputs are available, a percentage of 0.5 means that the rule heads predict for a
                                subset of `ceil(0.5 * 100) = 50` outputs. Must be in (0, 1) or 0, if the percentage
                                should be calculated based on the average label cardinality
        :return:                A `FixedPartialHeadConfig` that allows further configuration of the rule heads
        """
        if output_ratio != 0.0:
            assert_greater('output_ratio', output_ratio, 0)
            assert_less('output_ratio', output_ratio, 1)
        self.config_ptr.setOutputRatio(output_ratio)
        return self

    def get_min_outputs(self) -> int:
        """
        Returns the minimum number of outputs for which the rule heads predict.

        :return: The minimum number of outputs for which the rule heads predict
        """
        return self.config_ptr.getMinOutputs()

    def set_min_outputs(self, min_outputs: int) -> FixedPartialHeadConfig:
        """
        Sets the minimum number of outputs for which the rule heads should predict.

        :param min_outputs: The minimum number of outputs for which the rule heads should predict. Must be at least 2
        :return:            A `FixedPartialHeadConfig` that allows further configuration of the rule heads
        """
        assert_greater_or_equal('min_outputs', min_outputs, 2)
        self.config_ptr.setMinOutputs(min_outputs)
        return self

    def get_max_outputs(self) -> int:
        """
        Returns the maximum number of outputs for which the rule heads predict.

        :return: The maximum number of outputs for which the rule heads predict
        """
        return self.config_ptr.getMaxOutputs()

    def set_max_outputs(self, max_outputs: int) -> FixedPartialHeadConfig:
        """
        Sets the maximum number of outputs for which the rule heads should predict.

        :param max_outputs: The maximum number of outputs for which the rule heads should predict. Must be at least the
                            minimum number of outputs or 0, if the maximum number of outputs should not be restricted
        :return:            A `FixedPartialHeadConfig` that allows further configuration of the rule heads
        """
        if max_outputs != 0:
            assert_greater_or_equal('max_outputs', max_outputs, self.config_ptr.getMinOutputs())
        self.config_ptr.setMaxOutputs(max_outputs)
        return self


cdef class DynamicPartialHeadConfig:
    """
    Allows to configure partial rule heads that predict for a subset of the available outputs that is determined
    dynamically. Only those outputs for which the square of the predictive quality exceeds a certain threshold are
    included in a rule head.
    """

    def get_threshold(self) -> float:
        """
        Returns the threshold that affects for how many outputs the rule heads predict.

        :return: The threshold that affects for how many outputs the rule heads predict
        """
        return self.config_ptr.getThreshold()

    def set_threshold(self, threshold: float) -> DynamicPartialHeadConfig:
        """
        Sets the threshold that affects for how many outputs the rule heads should predict.

        :param threshold:   A threshold that affects for how many outputs the rule heads should predict. A smaller
                            threshold results in less outputs being selected. A greater threshold results in more
                            outputs being selected. E.g., a threshold of 0.2 means that a rule will only predict for a
                            output if the estimated predictive quality `q` for this particular output satisfies the
                            inequality `q^exponent > q_best^exponent * (1 - 0.2)`, where `q_best` is the best quality
                            among all outputs. Must be in (0, 1)
        :return:            A `DynamicPartialHeadConfig` that allows further configuration of the rule heads
        """
        assert_greater('threshold', threshold, 0)
        assert_less('threshold', threshold, 1)
        self.config_ptr.setThreshold(threshold)
        return self

    def get_exponent(self) -> float:
        """
        Sets the exponent that is used to weigh the estimated predictive quality for individual outputs.

        :return: The exponent that is used to weight the estimated predictive quality for individual outputs
        """
        return self.config_ptr.getExponent()

    def set_exponent(self, exponent: float) -> DynamicPartialHeadConfig:
        """
        Sets the exponent that should be used to weigh the estimated predictive quality for individual outputs.

        :param exponent:    An exponent that should be used to weigh the estimated predictive quality for individual
                            outputs. E.g., an exponent of 2 means that the estimated predictive quality `q` for a
                            particular output is weighed as `q^2`. Must be at least 1
        :return:            A `DynamicPartialHeadConfig` that allows further configuration of the rule heads
        """
        assert_greater_or_equal('exponent', exponent, 1)
        self.config_ptr.setExponent(exponent)
        return self
