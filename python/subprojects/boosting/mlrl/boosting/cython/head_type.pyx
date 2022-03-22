"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython._validation import assert_less, assert_greater, assert_greater_or_equal


cdef class FixedPartialHeadConfig:
    """
    Allows to configure partial rule heads that predict for a predefined number of labels.
    """

    def get_label_ratio(self) -> float:
        """
        Returns the percentage that specifies for how many labels the rule heads predict.

        :return: The percentage that specifies for how labels the rule heads predict or 0, if the percentage is
                 calculated based on the average label cardinality
        """
        return self.config_ptr.getLabelRatio()

    def set_label_ratio(self, label_ratio: float) -> FixedPartialHeadConfig:
        """
        Sets the percentage that specifies for how many labels the rule heads should predict.

        :param label_ratio: A percentage that specifies for how many labels the rule heads should predict, e.g., if 100
                            labels are available, a percentage of 0.5 means that the rule heads predict for a subset of
                            `ceil(0.5 * 100) = 50` labels. Must be in (0, 1) or 0, if the percentage should be
                            calculated based on the average label cardinality
        :return:            A `FixedPartialHeadConfig` that allows further configuration of the rule heads
        """
        if label_ratio != 0.0:
            assert_greater('label_ratio', label_ratio, 0)
            assert_less('label_ratio', label_ratio, 1)
        self.config_ptr.setLabelRatio(label_ratio)
        return self

    def get_min_labels(self) -> int:
        """
        Returns the minimum number of labels for which the rule heads predict.

        :return: The minimum number of labels for which the rule heads predict
        """
        return self.config_ptr.getMinLabels()

    def set_min_labels(self, min_labels: int) -> FixedPartialHeadConfig:
        """
        Sets the minimum number of labels for which the rule heads should predict.

        :param min_labels:  The minimum number of labels for which the rule heads should predict. Must be at least 2
        :return:            A `FixedPartialHeadConfig` that allows further configuration of the rule heads
        """
        assert_greater_or_equal('min_labels', min_labels, 2)
        self.config_ptr.setMinLabels(min_labels)
        return self

    def get_max_labels(self) -> int:
        """
        Returns the maximum number of labels for which the rule heads predict.

        :return: The maximum number of labels for which the rule heads predict
        """
        return self.config_ptr.getMaxLabels()

    def set_max_labels(self, max_labels: int) -> FixedPartialHeadConfig:
        """
        Sets the maximum number of labels for which the rule heads should predict.

        :param max_labels:  The maximum number of labels for which the rule heads should predict. Must be at least the
                            minimum number of labels or 0, if the maximum number of labels should not be restricted
        :return:            A `FixedPartialHeadConfig` that allows further configuration of the rule heads
        """
        if max_labels != 0:
            assert_greater_or_equal('max_labels', max_labels, self.config_ptr.getMinLabels())
        self.config_ptr.setMaxLabels(max_labels)
        return self
