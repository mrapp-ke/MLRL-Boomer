from mlrl.common.cython._validation import assert_at_least, assert_greater_or_equal, assert_less_or_equal, \
    assert_multiple, assert_not_none


cdef class SizeStoppingCriterionConfig:
    """
    A wrapper for the C++ class `SizeStoppingCriterionConfig`.
    """

    def set_max_rules(self, max_rules: int) -> SizeStoppingCriterionConfig:
        """
        Sets the maximum number of rules that should be induced.

        :param max_rules:   The maximum number of rules that should be induced. Must be at least 1
        :return:            A `SizeStoppingCriterionConfig` that allows further configuration of the stopping criterion
        """
        assert_at_least('max_rules', max_rules, 1)
        self.config_ptr.setMaxRules(max_rules)
        return self


cdef class TimeStoppingCriterionConfig:
    """
    A wrapper for the C++ class `TimeStoppingCriterionConfig`.
    """

    def set_time_limit(self, time_limit: int) -> TimeStoppingCriterionConfig:
        """
        Sets the time limit.

        :param time_limit:  The time limit in seconds. Must be at least 1
        :return:            A `TimeStoppingCriterionConfig` that allows further configuration of the stopping criterion
        """
        assert_at_least('time_limit', time_limit, 1)
        self.config_ptr.setTimeLimit(time_limit)
        return self


cdef class MeasureStoppingCriterionConfig:
    """
    A wrapper for the C++ class `MeasureStoppingCriterionConfig`.
    """

    def set_aggregation_function(self, aggregation_function: AggregationFunction) -> MeasureStoppingCriterionConfig:
        """
        Sets the type of the aggregation function that should be used to aggregate the values that are stored in a
        buffer.

        :param aggregation_function:    A value of the enum `AggregationFunction` that specifies the type of the
                                        aggregation function that should be used to aggregate the values that are stored
                                        in a buffer
        :return:                        A reference to an object of type `MeasureStoppingCriterionConfig` that allows
                                        further configuration of the stopping criterion
        """
        assert_not_none('aggregation_function', aggregation_function)
        cdef uint32 enum_value = aggregation_function
        self.config_ptr.setAggregationFunction(<AggregationFunctionImpl>enum_value)
        return self

    def set_min_rules(self, min_rules: int) -> MeasureStoppingCriterionConfig:
        """
        Sets the minimum number of rules that must have been learned until the induction of rules might be stopped.

        :param min_rules:   The minimum number of rules that must have been learned until the induction of rules might
                            be stopped. Must be at least 1
        :return:            A `MeasureStoppingCriterionConfig` that allows further configuration of the stopping
                            criterion
        """
        assert_at_least('min_rules', min_rules, 1)
        self.config_ptr.setMinRules(min_rules)
        return self

    def set_update_interval(self, update_interval: int) -> MeasureStoppingCriterionConfig:
        """
        Sets the interval that should be used to update the quality of the current model.

        :param update_interval: The interval that should be used to update the quality of the current model, e.g., a
         *                      value of 5 means that the model quality is assessed every 5 rules. Must be at least 1
        :return:                A `MeasureStoppingCriterionConfig` that allows further configuration of the stopping
                                criterion
        """
        assert_greater_or_equal('update_interval', update_interval, 1)
        self.config_ptr.setUpdateInterval(update_interval)
        return self

    def set_stop_interval(self, stop_interval: int) -> MeasureStoppingCriterionConfig:
        """
        Sets the interval that should be used to decide whether the induction of rules should be stopped.

        :param stop_interval:   The interval that should be used to decide whether the induction of rules should be
                                stopped, e.g., a value of 10 means that the rule induction might be stopped after 10,
                                20, ... rules. Must be a multiple of the update interval
        :return:                A `MeasureStoppingCriterionConfig` that allows further configuration of the stopping
                                criterion
        """
        assert_multiple('stop_interval', stop_interval, self.config_ptr.getUpdateInterval())
        self.config_ptr.setStopInterval(stop_interval)
        return self

    def set_num_past(self, num_past: int) -> MeasureStoppingCriterionConfig:
        """
        Sets the number of quality scores of past iterations that should be stored in a buffer.

        :param num_past:    The number of quality scores of past iterations that should be be stored in a buffer. Must
                            be at least 1
        :return:            A `MeasureStoppingCriterionConfig` that allows further configuration of the stopping
                            criterion
        """
        assert_at_least('num_past', num_past, 1)
        self.config_ptr.setNumPast(num_past)
        return self

    def set_num_current(self, num_current: int) -> MeasureStoppingCriterionConfig:
        """
        Sets the number of quality scores of the most recent iterations that should be stored in a buffer.

        :param num_current: The number of quality scores of the most recent iterations that should be stored in a
                            buffer. Must be at least 1
        :return:            A `MeasureStoppingCriterionConfig` that allows further configuration of the stopping
                            criterion
        """
        assert_at_least('num_current', num_current, 1)
        self.config_ptr.setNumCurrent(num_current)
        return self

    def set_min_improvement(self, min_improvement: float) -> MeasureStoppingCriterionConfig:
        """
        Sets the minimum improvement that must be reached for the rule induction to be continued.

        :param min_improvement: The minimum improvement in percent that must be reached for the rule induction to be
                                continued. Must be in [0, 1]
        :return:                A `MeasureStoppingCriterionConfig` that allows further configuration of the stopping
                                criterion
        """
        assert_greater_or_equal('min_improvement', min_improvement, 0)
        assert_less_or_equal('min_improvement', min_improvement, 1)
        self.config_ptr.setMinImprovement(min_improvement)
        return self

    def set_force_stop(self, force_stop: bool) -> MeasureStoppingCriterionConfig:
        """
        Sets whether the induction of rules should be forced to be stopped, if the stopping criterion is met.

        :param force_stop:  True, if the induction of rules should be forced to be stopped, if the stopping criterion is
                            met, False, if only the time of stopping should be stored
        :return:            A `MeasureStoppingCriterionConfig` that allows further configuration of the stopping
                            criterion
        """
        self.config_ptr.setForceStop(force_stop)
        return self
