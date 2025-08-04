"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utilities that ease the configuration of rule learning algorithms.
"""
import logging as log

from abc import ABC, abstractmethod
from typing import Dict, Optional, Set, override

from mlrl.common.cython.learner import BeamSearchTopDownRuleInductionMixin, EqualFrequencyFeatureBinningMixin, \
    EqualWidthFeatureBinningMixin, FeatureSamplingWithoutReplacementMixin, GreedyTopDownRuleInductionMixin, \
    InstanceSamplingWithoutReplacementMixin, InstanceSamplingWithReplacementMixin, IrepRulePruningMixin, \
    NoFeatureBinningMixin, NoFeatureSamplingMixin, NoGlobalPruningMixin, NoInstanceSamplingMixin, \
    NoParallelPredictionMixin, NoParallelRuleRefinementMixin, NoParallelStatisticUpdateMixin, \
    NoPartitionSamplingMixin, NoRulePruningMixin, NoSequentialPostOptimizationMixin, NoSizeStoppingCriterionMixin, \
    NoTimeStoppingCriterionMixin, ParallelPredictionMixin, ParallelRuleRefinementMixin, ParallelStatisticUpdateMixin, \
    PostPruningMixin, PrePruningMixin, RandomBiPartitionSamplingMixin, RNGMixin, RoundRobinOutputSamplingMixin, \
    SequentialPostOptimizationMixin, SizeStoppingCriterionMixin, TimeStoppingCriterionMixin
from mlrl.common.cython.learner_classification import ExampleWiseStratifiedBiPartitionSamplingMixin, \
    ExampleWiseStratifiedInstanceSamplingMixin, OutputWiseStratifiedBiPartitionSamplingMixin, \
    OutputWiseStratifiedInstanceSamplingMixin
from mlrl.common.cython.package_info import get_num_cpu_cores, is_multi_threading_support_enabled
from mlrl.common.cython.stopping_criterion import AggregationFunction

from mlrl.util.cli import NONE, Argument, SetArgument
from mlrl.util.options import BooleanOption, Options, parse_param, parse_param_and_options

OPTION_RESAMPLE_FEATURES = 'resample_features'

SAMPLING_WITH_REPLACEMENT = 'with-replacement'

SAMPLING_WITHOUT_REPLACEMENT = 'without-replacement'

SAMPLING_STRATIFIED_OUTPUT_WISE = 'stratified-output-wise'

SAMPLING_STRATIFIED_EXAMPLE_WISE = 'stratified-example-wise'

OPTION_SAMPLE_SIZE = 'sample_size'

OPTION_MIN_SAMPLES = 'min_samples'

OPTION_MAX_SAMPLES = 'max_samples'

BINNING_EQUAL_FREQUENCY = 'equal-frequency'

BINNING_EQUAL_WIDTH = 'equal-width'

OPTION_BIN_RATIO = 'bin_ratio'

OPTION_MIN_BINS = 'min_bins'

OPTION_MAX_BINS = 'max_bins'

OPTION_NUM_PREFERRED_THREADS = 'num_preferred_threads'

OPTION_USE_HOLDOUT_SET = 'use_holdout_set'


class Parameter(ABC):
    """
    An abstract base class for all parameters of a rule learning algorithm.
    """

    def __init__(self, name: str, description: str):
        """
        :param name:        The name of the parameter
        :param description: A textual description of the parameter
        """
        self.name = name
        self.description = description

    @abstractmethod
    def configure(self, config, value):
        """
        Configures a rule learner depending on this parameter.

        :param config:  The configuration to be modified
        :param value:   The value to be set
        """

    @abstractmethod
    def as_argument(self, config_type: type) -> Optional[Argument]:
        """
        Creates and returns an `Argument` from this parameter, if it is supported by a configuration of a specific type.

        :param config_type: The type of the configuration
        :return:            The `Argument` that has been created or None, if it is not supported by a configuration of
                            the given type
        """

    @property
    def argument_name(self) -> str:
        """
        The name of a command line argument that corresponds to the parameter.
        """
        return '--' + self.name.replace('_', '-')

    @override
    def __eq__(self, other):
        return self.name == other.name

    @override
    def __hash__(self):
        return hash(self.name)


class NominalParameter(Parameter, ABC):
    """
    An abstract base class for all nominal parameters of a rule learning algorithm that allow to set one out of a set of
    predefined values.
    """

    class Value:
        """
        A value that can be set for a nominal parameter.
        """

        def __init__(self, name: str, mixin: type, options: Optional[Set[str]], description: Optional[str]):
            """
            :param name:        The name of the value
            :param mixin:       The type of the mixin that must be implemented by a rule learner to support this value
            :param options:     A set that contains the names of additional options that may be specified or None, if no
                                additional options are available
            :param description: A textual description of the value
            """
            self.name = name
            self.mixin = mixin
            self.options = options
            self.description = description

        @override
        def __eq__(self, other):
            return self.name == other.name

        @override
        def __hash__(self):
            return hash(self.name)

    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.values: Dict[str, NominalParameter.Value] = {}

    def add_value(self, name: str, mixin: type, options: Optional[Set[str]] = None, description: Optional[str] = None):
        """
        Adds a new value to the parameter.

        :param name:        The name of the value to be added
        :param mixin:       The type of the mixin that must be implemented by a rule learner to support the value
        :param options:     A set that contains the names of additional options that may be specified or None, if no
                            additional options are available
        :param description: A textual description of the value
        :return:            The parameter itself
        """
        self.values[name] = NominalParameter.Value(name=name, mixin=mixin, options=options, description=description)
        return self

    @abstractmethod
    def _configure(self, config, value: str, options: Options):
        """
        Must be implemented by subclasses in order to configure a rule learner depending on the specified nominal value.
        
        :param config:  The configuration to be modified
        :param value:   The nominal value to be set
        :param options: Additional options that have eventually been specified
        """

    def __get_supported_values(self, config_type: type) -> Set[str] | Dict[str, Options]:
        num_options = 0
        supported_values = {}

        for parameter_value in self.values.values():
            if issubclass(config_type, parameter_value.mixin):
                options = parameter_value.options
                num_options += len(options) if options else 0
                supported_values[parameter_value.name] = options if options else set()

        return set(supported_values.keys()) if num_options == 0 else supported_values

    @override
    def configure(self, config, value):
        supported_values = self.__get_supported_values(type(config))

        if supported_values:
            value = str(value)

            if isinstance(supported_values, dict):
                value, options = parse_param_and_options(self.name, value, supported_values)
            else:
                value = parse_param(self.name, value, supported_values)
                options = None

            self._configure(config, value, options if options else Options())

    @override
    def as_argument(self, config_type: type) -> Optional[Argument]:
        supported_values = self.__get_supported_values(config_type)

        if supported_values:
            description = self.description

            for supported_value in supported_values:
                value_description = self.values[supported_value].description

                if value_description:
                    description += ' ' + value_description + '.'

            return SetArgument(self.argument_name, values=supported_values, description=description)

        return None


class NumericalParameter(Parameter, ABC):
    """
    An abstract base class for all parameters of a rule learning algorithm that allow to set a numerical value.
    """

    def __init__(self, name: str, description: str, mixin: type, numeric_type: type):
        """
        :param mixin:           The type of the mixin that must be implemented by a rule learner to support the
                                parameter
        :param numeric_type:    The type of the numerical value
        """
        super().__init__(name=name, description=description)
        self.mixin = mixin
        self.numeric_type = numeric_type

    @abstractmethod
    def _configure(self, config, value):
        """
        Must be implemented by subclasses in order to configure a rule learner depending on the specified numerical
        value.
        
        :param config:  The configuration to be modified
        :param value:   The numerical value to be set
        """

    def __is_supported(self, config_type: type):
        return issubclass(config_type, self.mixin)

    @override
    def configure(self, config, value):
        if self.__is_supported(type(config)):
            self._configure(config, self.numeric_type(value))

    @override
    def as_argument(self, config_type: type) -> Optional[Argument]:
        if self.__is_supported(config_type):
            return Argument(self.argument_name, type=self.numeric_type, help=self.description)
        return None


class IntParameter(NumericalParameter, ABC):
    """
    An abstract base class for all parameters of a rule learning algorithm that allow to set an integer value.
    """

    def __init__(self, name: str, description: str, mixin: type):
        """
        :param mixin: The type of the mixin that must be implemented by a rule learner to support the parameter
        """
        super().__init__(name=name, description=description, mixin=mixin, numeric_type=int)


class FloatParameter(NumericalParameter, ABC):
    """
    An abstract base class for all parameters of a rule learning algorithm that allow to set a floating point value.
    """

    def __init__(self, name: str, description: str, mixin: type):
        super().__init__(name=name, description=description, mixin=mixin, numeric_type=float)


class RandomStateParameter(IntParameter):
    """
    A parameter that allows to configure the seed to be used by random number generators.
    """

    def __init__(self):
        super().__init__(name='random_state',
                         description='The seed to be used by random number generators. Must be at least 1.',
                         mixin=RNGMixin)

    @override
    def _configure(self, config, value):
        config.use_rng().set_random_state(value)


class RuleInductionParameter(NominalParameter):
    """
    A parameter that allows to configure the algorithm to be used for the induction of individual rules.
    """

    RULE_INDUCTION_TOP_DOWN_GREEDY = 'top-down-greedy'

    OPTION_MIN_COVERAGE = 'min_coverage'

    OPTION_MIN_SUPPORT = 'min_support'

    OPTION_MAX_CONDITIONS = 'max_conditions'

    OPTION_MAX_HEAD_REFINEMENTS = 'max_head_refinements'

    OPTION_RECALCULATE_PREDICTIONS = 'recalculate_predictions'

    RULE_INDUCTION_TOP_DOWN_BEAM_SEARCH = 'top-down-beam-search'

    OPTION_BEAM_WIDTH = 'beam_width'

    def __init__(self):
        super().__init__(name='rule_induction',
                         description='The name of the algorithm to be used for the induction of individual rules')
        self.add_value(name=self.RULE_INDUCTION_TOP_DOWN_GREEDY,
                       mixin=GreedyTopDownRuleInductionMixin,
                       options={
                           self.OPTION_MIN_COVERAGE, self.OPTION_MIN_SUPPORT, self.OPTION_MAX_CONDITIONS,
                           self.OPTION_MAX_HEAD_REFINEMENTS, self.OPTION_RECALCULATE_PREDICTIONS
                       })
        self.add_value(name=self.RULE_INDUCTION_TOP_DOWN_BEAM_SEARCH,
                       mixin=BeamSearchTopDownRuleInductionMixin,
                       options={
                           self.OPTION_MIN_COVERAGE, self.OPTION_MIN_SUPPORT, self.OPTION_MAX_CONDITIONS,
                           self.OPTION_MAX_HEAD_REFINEMENTS, self.OPTION_RECALCULATE_PREDICTIONS,
                           self.OPTION_BEAM_WIDTH, OPTION_RESAMPLE_FEATURES
                       })

    @override
    def _configure(self, config, value: str, options: Options):
        if value == self.RULE_INDUCTION_TOP_DOWN_GREEDY:
            conf = config.use_greedy_top_down_rule_induction()
            conf.set_min_coverage(options.get_int(self.OPTION_MIN_COVERAGE, conf.get_min_coverage()))
            conf.set_min_support(options.get_float(self.OPTION_MIN_SUPPORT, conf.get_min_support()))
            conf.set_max_conditions(options.get_int(self.OPTION_MAX_CONDITIONS, conf.get_max_conditions()))
            conf.set_max_head_refinements(
                options.get_int(self.OPTION_MAX_HEAD_REFINEMENTS, conf.get_max_head_refinements()))
            conf.set_recalculate_predictions(
                options.get_bool(self.OPTION_RECALCULATE_PREDICTIONS, conf.are_predictions_recalculated()))
        elif value == self.RULE_INDUCTION_TOP_DOWN_BEAM_SEARCH:
            conf = config.use_beam_search_top_down_rule_induction()
            conf.set_min_coverage(options.get_int(self.OPTION_MIN_COVERAGE, conf.get_min_coverage()))
            conf.set_min_support(options.get_float(self.OPTION_MIN_SUPPORT, conf.get_min_support()))
            conf.set_max_conditions(options.get_int(self.OPTION_MAX_CONDITIONS, conf.get_max_conditions()))
            conf.set_max_head_refinements(
                options.get_int(self.OPTION_MAX_HEAD_REFINEMENTS, conf.get_max_head_refinements()))
            conf.set_recalculate_predictions(
                options.get_bool(self.OPTION_RECALCULATE_PREDICTIONS, conf.are_predictions_recalculated()))
            conf.set_beam_width(options.get_int(self.OPTION_BEAM_WIDTH, conf.get_beam_width()))
            conf.set_resample_features(options.get_bool(OPTION_RESAMPLE_FEATURES, conf.are_features_resampled()))


class FeatureBinningParameter(NominalParameter):
    """
    A parameter that allows to configure the strategy to be used for feature binning.
    """

    def __init__(self):
        super().__init__(name='feature_binning', description='The name of the strategy to be used for feature binning')
        self.add_value(name=NONE, mixin=NoFeatureBinningMixin)
        self.add_value(name=BINNING_EQUAL_FREQUENCY,
                       mixin=EqualFrequencyFeatureBinningMixin,
                       options={OPTION_BIN_RATIO, OPTION_MIN_BINS, OPTION_MAX_BINS})
        self.add_value(name=BINNING_EQUAL_WIDTH,
                       mixin=EqualWidthFeatureBinningMixin,
                       options={OPTION_BIN_RATIO, OPTION_MIN_BINS, OPTION_MAX_BINS})

    @override
    def _configure(self, config, value: str, options: Options):
        if value == NONE:
            config.use_no_feature_binning()
        elif value == BINNING_EQUAL_FREQUENCY:
            conf = config.use_equal_frequency_feature_binning()
            conf.set_bin_ratio(options.get_float(OPTION_BIN_RATIO, conf.get_bin_ratio()))
            conf.set_min_bins(options.get_int(OPTION_MIN_BINS, conf.get_min_bins()))
            conf.set_max_bins(options.get_int(OPTION_MAX_BINS, conf.get_max_bins()))
        elif value == BINNING_EQUAL_WIDTH:
            conf = config.use_equal_width_feature_binning()
            conf.set_bin_ratio(options.get_float(OPTION_BIN_RATIO, conf.get_bin_ratio()))
            conf.set_min_bins(options.get_int(OPTION_MIN_BINS, conf.get_min_bins()))
            conf.set_max_bins(options.get_int(OPTION_MAX_BINS, conf.get_max_bins()))


class OutputSamplingParameter(NominalParameter):
    """
    A parameter that allows to configure the strategy to be used for output sampling.
    """

    OUTPUT_SAMPLING_ROUND_ROBIN = 'round-robin'

    def __init__(self):
        super().__init__(name='output_sampling', description='The name of the strategy to be used for output sampling')
        self.add_value(name=NONE, mixin=NoFeatureSamplingMixin)
        self.add_value(name=SAMPLING_WITHOUT_REPLACEMENT,
                       mixin=FeatureSamplingWithoutReplacementMixin,
                       options={OPTION_SAMPLE_SIZE, OPTION_MIN_SAMPLES, OPTION_MAX_SAMPLES})
        self.add_value(name=self.OUTPUT_SAMPLING_ROUND_ROBIN, mixin=RoundRobinOutputSamplingMixin)

    @override
    def _configure(self, config, value: str, options: Options):
        if value == NONE:
            config.use_no_output_sampling()
        elif value == SAMPLING_WITHOUT_REPLACEMENT:
            conf = config.use_output_sampling_without_replacement()
            conf.set_sample_size(options.get_float(OPTION_SAMPLE_SIZE, conf.get_sample_size()))
            conf.set_min_samples(options.get_int(OPTION_MIN_SAMPLES, conf.get_min_samples()))
            conf.set_max_samples(options.get_int(OPTION_MAX_SAMPLES, conf.get_max_samples()))
        elif value == self.OUTPUT_SAMPLING_ROUND_ROBIN:
            config.use_round_robin_output_sampling()


class InstanceSamplingParameter(NominalParameter):
    """
    A parameter that allows to configure the strategy to be used for instance sampling.
    """

    def __init__(self):
        super().__init__(name='instance_sampling',
                         description='The name of the strategy to be used for instance sampling')
        self.add_value(name=NONE, mixin=NoInstanceSamplingMixin)
        self.add_value(name=SAMPLING_WITH_REPLACEMENT,
                       mixin=InstanceSamplingWithReplacementMixin,
                       options={OPTION_SAMPLE_SIZE, OPTION_MIN_SAMPLES, OPTION_MAX_SAMPLES})
        self.add_value(name=SAMPLING_WITHOUT_REPLACEMENT,
                       mixin=InstanceSamplingWithoutReplacementMixin,
                       options={OPTION_SAMPLE_SIZE, OPTION_MIN_SAMPLES, OPTION_MAX_SAMPLES})
        self.add_value(name=SAMPLING_STRATIFIED_OUTPUT_WISE,
                       mixin=OutputWiseStratifiedInstanceSamplingMixin,
                       options={OPTION_SAMPLE_SIZE, OPTION_MIN_SAMPLES, OPTION_MAX_SAMPLES})
        self.add_value(name=SAMPLING_STRATIFIED_EXAMPLE_WISE,
                       mixin=ExampleWiseStratifiedInstanceSamplingMixin,
                       options={OPTION_SAMPLE_SIZE, OPTION_MIN_SAMPLES, OPTION_MAX_SAMPLES})

    @override
    def _configure(self, config, value: str, options: Options):
        if value == NONE:
            config.use_no_instance_sampling()
        elif value == SAMPLING_WITH_REPLACEMENT:
            conf = config.use_instance_sampling_with_replacement()
            conf.set_sample_size(options.get_float(OPTION_SAMPLE_SIZE, conf.get_sample_size()))
            conf.set_min_samples(options.get_int(OPTION_MIN_SAMPLES, conf.get_min_samples()))
            conf.set_max_samples(options.get_int(OPTION_MAX_SAMPLES, conf.get_max_samples()))
        elif value == SAMPLING_WITHOUT_REPLACEMENT:
            conf = config.use_instance_sampling_without_replacement()
            conf.set_sample_size(options.get_float(OPTION_SAMPLE_SIZE, conf.get_sample_size()))
            conf.set_min_samples(options.get_int(OPTION_MIN_SAMPLES, conf.get_min_samples()))
            conf.set_max_samples(options.get_int(OPTION_MAX_SAMPLES, conf.get_max_samples()))
        elif value == SAMPLING_STRATIFIED_OUTPUT_WISE:
            conf = config.use_output_wise_stratified_instance_sampling()
            conf.set_sample_size(options.get_float(OPTION_SAMPLE_SIZE, conf.get_sample_size()))
            conf.set_min_samples(options.get_int(OPTION_MIN_SAMPLES, conf.get_min_samples()))
            conf.set_max_samples(options.get_int(OPTION_MAX_SAMPLES, conf.get_max_samples()))
        elif value == SAMPLING_STRATIFIED_EXAMPLE_WISE:
            conf = config.use_example_wise_stratified_instance_sampling()
            conf.set_sample_size(options.get_float(OPTION_SAMPLE_SIZE, conf.get_sample_size()))
            conf.set_min_samples(options.get_int(OPTION_MIN_SAMPLES, conf.get_min_samples()))
            conf.set_max_samples(options.get_int(OPTION_MAX_SAMPLES, conf.get_max_samples()))


class FeatureSamplingParameter(NominalParameter):
    """
    A parameter that allows to configure the strategy to be used for feature sampling.
    """

    OPTION_NUM_RETAINED = 'num_retained'

    def __init__(self):
        super().__init__(name='feature_sampling',
                         description='The name of the strategy to be used for feature sampling')
        self.add_value(name=NONE, mixin=NoFeatureSamplingMixin)
        self.add_value(name=SAMPLING_WITHOUT_REPLACEMENT,
                       mixin=FeatureSamplingWithoutReplacementMixin,
                       options={OPTION_SAMPLE_SIZE, OPTION_MIN_SAMPLES, OPTION_MAX_SAMPLES, self.OPTION_NUM_RETAINED})

    @override
    def _configure(self, config, value: str, options: Options):
        if value == NONE:
            config.use_no_feature_sampling()
        elif value == SAMPLING_WITHOUT_REPLACEMENT:
            conf = config.use_feature_sampling_without_replacement()
            conf.set_sample_size(options.get_float(OPTION_SAMPLE_SIZE, conf.get_sample_size()))
            conf.set_min_samples(options.get_int(OPTION_MIN_SAMPLES, conf.get_min_samples()))
            conf.set_max_samples(options.get_int(OPTION_MAX_SAMPLES, conf.get_max_samples()))
            conf.set_num_retained(options.get_int(self.OPTION_NUM_RETAINED, conf.get_num_retained()))


class PartitionSamplingParameter(NominalParameter):
    """
    A parameter that allows to configure the strategy to be used for creating a holdout set.
    """

    PARTITION_SAMPLING_RANDOM = 'random'

    OPTION_HOLDOUT_SET_SIZE = 'holdout_set_size'

    def __init__(self):
        super().__init__(name='holdout', description='The name of the strategy to be used for creating a holdout set')
        self.add_value(name=NONE, mixin=NoPartitionSamplingMixin)
        self.add_value(name=self.PARTITION_SAMPLING_RANDOM,
                       mixin=RandomBiPartitionSamplingMixin,
                       options={self.OPTION_HOLDOUT_SET_SIZE})
        self.add_value(name=SAMPLING_STRATIFIED_OUTPUT_WISE,
                       mixin=OutputWiseStratifiedBiPartitionSamplingMixin,
                       options={self.OPTION_HOLDOUT_SET_SIZE})
        self.add_value(name=SAMPLING_STRATIFIED_EXAMPLE_WISE,
                       mixin=ExampleWiseStratifiedBiPartitionSamplingMixin,
                       options={self.OPTION_HOLDOUT_SET_SIZE})

    @override
    def _configure(self, config, value: str, options: Options):
        if value == NONE:
            config.use_no_partition_sampling()
        elif value == self.PARTITION_SAMPLING_RANDOM:
            conf = config.use_random_bi_partition_sampling()
            conf.set_holdout_set_size(options.get_float(self.OPTION_HOLDOUT_SET_SIZE, conf.get_holdout_set_size()))
        elif value == SAMPLING_STRATIFIED_OUTPUT_WISE:
            conf = config.use_output_wise_stratified_bi_partition_sampling()
            conf.set_holdout_set_size(options.get_float(self.OPTION_HOLDOUT_SET_SIZE, conf.get_holdout_set_size()))
        elif value == SAMPLING_STRATIFIED_EXAMPLE_WISE:
            conf = config.use_example_wise_stratified_bi_partition_sampling()
            conf.set_holdout_set_size(options.get_float(self.OPTION_HOLDOUT_SET_SIZE, conf.get_holdout_set_size()))


class GlobalPruningParameter(NominalParameter):
    """
    A parameter that allows to configure the strategy to be used for pruning entire rules.
    """

    GLOBAL_PRUNING_POST = 'post-pruning'

    OPTION_REMOVE_UNUSED_RULES = 'remove_unused_rules'

    OPTION_MIN_RULES = 'min_rules'

    OPTION_INTERVAL = 'interval'

    GLOBAL_PRUNING_PRE = 'pre-pruning'

    OPTION_AGGREGATION_FUNCTION = 'aggregation'

    OPTION_UPDATE_INTERVAL = 'update_interval'

    OPTION_STOP_INTERVAL = 'stop_interval'

    OPTION_NUM_PAST = 'num_past'

    OPTION_NUM_RECENT = 'num_recent'

    OPTION_MIN_IMPROVEMENT = 'min_improvement'

    AGGREGATION_FUNCTION_MIN = 'min'

    AGGREGATION_FUNCTION_MAX = 'max'

    AGGREGATION_FUNCTION_ARITHMETIC_MEAN = 'avg'

    def __init__(self):
        super().__init__(name='global_pruning',
                         description='The name of the strategy to be used for pruning entire rules')
        self.add_value(name=NONE, mixin=NoGlobalPruningMixin)
        self.add_value(name=self.GLOBAL_PRUNING_POST,
                       mixin=PostPruningMixin,
                       options={
                           OPTION_USE_HOLDOUT_SET, self.OPTION_REMOVE_UNUSED_RULES, self.OPTION_MIN_RULES,
                           self.OPTION_INTERVAL
                       })
        self.add_value(name=self.GLOBAL_PRUNING_PRE,
                       mixin=PrePruningMixin,
                       options={
                           OPTION_USE_HOLDOUT_SET, self.OPTION_REMOVE_UNUSED_RULES, self.OPTION_MIN_RULES,
                           self.OPTION_AGGREGATION_FUNCTION, self.OPTION_UPDATE_INTERVAL, self.OPTION_STOP_INTERVAL,
                           self.OPTION_NUM_PAST, self.OPTION_NUM_RECENT, self.OPTION_MIN_IMPROVEMENT
                       })

    def __create_aggregation_function(self, aggregation_function: str) -> AggregationFunction:
        value = parse_param(
            self.OPTION_AGGREGATION_FUNCTION, aggregation_function,
            {self.AGGREGATION_FUNCTION_MIN, self.AGGREGATION_FUNCTION_MAX, self.AGGREGATION_FUNCTION_ARITHMETIC_MEAN})

        if value == self.AGGREGATION_FUNCTION_MIN:
            return AggregationFunction.MIN
        if value == self.AGGREGATION_FUNCTION_MAX:
            return AggregationFunction.MAX
        return AggregationFunction.ARITHMETIC_MEAN

    @override
    def _configure(self, config, value: str, options: Options):
        if value == NONE:
            config.use_no_global_pruning()
        elif value == self.GLOBAL_PRUNING_POST:
            conf = config.use_global_post_pruning()
            conf.set_use_holdout_set(options.get_bool(OPTION_USE_HOLDOUT_SET, conf.is_holdout_set_used()))
            conf.set_remove_unused_rules(
                options.get_bool(self.OPTION_REMOVE_UNUSED_RULES, conf.is_remove_unused_rules()))
            conf.set_min_rules(options.get_int(self.OPTION_MIN_RULES, conf.get_min_rules()))
            conf.set_interval(options.get_int(self.OPTION_INTERVAL, conf.get_interval()))
        elif value == self.GLOBAL_PRUNING_PRE:
            conf = config.use_global_pre_pruning()
            conf.set_use_holdout_set(options.get_bool(OPTION_USE_HOLDOUT_SET, conf.is_holdout_set_used()))
            conf.set_remove_unused_rules(
                options.get_bool(self.OPTION_REMOVE_UNUSED_RULES, conf.is_remove_unused_rules()))
            conf.set_min_rules(options.get_int(self.OPTION_MIN_RULES, conf.get_min_rules()))
            aggregation_function = options.get_string(self.OPTION_AGGREGATION_FUNCTION, None)
            conf.set_aggregation_function(
                self.__create_aggregation_function(aggregation_function) if aggregation_function else conf
                .get_aggregation_function())
            conf.set_update_interval(options.get_int(self.OPTION_UPDATE_INTERVAL, conf.get_update_interval()))
            conf.set_stop_interval(options.get_int(self.OPTION_STOP_INTERVAL, conf.get_stop_interval()))
            conf.set_num_past(options.get_int(self.OPTION_NUM_PAST, conf.get_num_past()))
            conf.set_num_current(options.get_int(self.OPTION_NUM_RECENT, conf.get_num_current()))
            conf.set_min_improvement(options.get_float(self.OPTION_MIN_IMPROVEMENT, conf.get_min_improvement()))


class RulePruningParameter(NominalParameter):
    """
    A parameter that allows to configure the strategy to be used for pruning individual rules.
    """

    RULE_PRUNING_IREP = 'irep'

    def __init__(self):
        super().__init__(name='rule_pruning',
                         description='The name of the strategy to be used for pruning individual rules')
        self.add_value(name=NONE, mixin=NoRulePruningMixin)
        self.add_value(name=self.RULE_PRUNING_IREP, mixin=IrepRulePruningMixin)

    @override
    def _configure(self, config, value: str, _: Options):
        if value == NONE:
            config.use_no_rule_pruning()
        elif value == self.RULE_PRUNING_IREP:
            config.use_irep_rule_pruning()


class ParallelRuleRefinementParameter(NominalParameter):
    """
    A parameter that allows to configure whether potential refinements of rules should be searched for in parallel or
    not.
    """

    def __init__(self):
        super().__init__(name='parallel_rule_refinement',
                         description='Whether potential refinements of rules should be searched for in parallel or not')
        self.add_value(name=BooleanOption.FALSE, mixin=NoParallelRuleRefinementMixin)
        self.add_value(name=BooleanOption.TRUE,
                       mixin=ParallelRuleRefinementMixin,
                       options={OPTION_NUM_PREFERRED_THREADS})

    @override
    def _configure(self, config, value: str, options: Options):
        if value == BooleanOption.FALSE:
            config.use_no_parallel_rule_refinement()
        else:
            conf = config.use_parallel_rule_refinement()
            num_preferred_threads = options.get_int(OPTION_NUM_PREFERRED_THREADS, conf.get_num_preferred_threads())
            conf.set_num_preferred_threads(num_preferred_threads)
            if num_preferred_threads > 1 and not is_multi_threading_support_enabled():
                log.warning('%s threads should be used for rule refinement, but multi-threading support is disabled',
                            num_preferred_threads)
            elif num_preferred_threads > get_num_cpu_cores():
                log.warning('%s threads should be used for rule refinement, but only %s CPU cores are available',
                            num_preferred_threads, get_num_cpu_cores())


class ParallelStatisticUpdateParameter(NominalParameter):
    """
    A parameter that allows to configure whether the statistics for different examples should be updated in parallel or
    not.
    """

    def __init__(self):
        super().__init__(
            name='parallel_statistic_update',
            description='Whether the statistics for different examples should be updated in parallel or not')
        self.add_value(name=BooleanOption.FALSE, mixin=NoParallelStatisticUpdateMixin)
        self.add_value(name=BooleanOption.TRUE,
                       mixin=ParallelStatisticUpdateMixin,
                       options={OPTION_NUM_PREFERRED_THREADS})

    @override
    def _configure(self, config, value: str, options: Options):
        if value == BooleanOption.FALSE:
            config.use_no_parallel_statistic_update()
        else:
            conf = config.use_parallel_statistic_update()
            num_preferred_threads = options.get_int(OPTION_NUM_PREFERRED_THREADS, conf.get_num_preferred_threads())
            conf.set_num_preferred_threads(num_preferred_threads)
            if num_preferred_threads > 1 and not is_multi_threading_support_enabled():
                log.warning('%s threads should be used for statistic updates, but multi-threading support is disabled',
                            num_preferred_threads)
            elif num_preferred_threads > get_num_cpu_cores():
                log.warning('%s threads should be used for statistic updates, but only %s CPU cores are available',
                            num_preferred_threads, get_num_cpu_cores())


class ParallelPredictionParameter(NominalParameter):
    """
    A parameter that allows to configure whether predictions for different examples should be obtained in parallel or
    not.
    """

    def __init__(self):
        super().__init__(name='parallel_prediction',
                         description='Whether predictions for different examples should be obtained in parallel or not')
        self.add_value(name=BooleanOption.FALSE, mixin=NoParallelPredictionMixin)
        self.add_value(name=BooleanOption.TRUE, mixin=ParallelPredictionMixin, options={OPTION_NUM_PREFERRED_THREADS})

    @override
    def _configure(self, config, value: str, options: Options):
        if value == BooleanOption.FALSE:
            config.use_no_parallel_prediction()
        else:
            conf = config.use_parallel_prediction()
            num_preferred_threads = options.get_int(OPTION_NUM_PREFERRED_THREADS, conf.get_num_preferred_threads())
            conf.set_num_preferred_threads(num_preferred_threads)
            if num_preferred_threads > 1 and not is_multi_threading_support_enabled():
                log.warning('%s threads should be used for prediction, but multi-threading support is disabled',
                            num_preferred_threads)
            elif num_preferred_threads > get_num_cpu_cores():
                log.warning('%s threads should be used for prediction, but only %s CPU cores are available',
                            num_preferred_threads, get_num_cpu_cores())


class SizeStoppingCriterionParameter(IntParameter):
    """
    A parameter that allows to configure the maximum number of rules to be induced.
    """

    def __init__(self):
        super().__init__(
            name='max_rules',
            description='The maximum number of rules to be induced. Must be at least 1 or 0, if the number of rules '
            + 'should not be restricted',
            mixin=SizeStoppingCriterionMixin)

    @override
    def _configure(self, config, value):
        if value == 0 and issubclass(type(config), NoSizeStoppingCriterionMixin):
            config.use_no_size_stopping_criterion()
        else:
            config.use_size_stopping_criterion().set_max_rules(value)


class TimeStoppingCriterionParameter(IntParameter):
    """
    A parameter that allows to configure the duration in seconds after which the induction of rules should be canceled.
    """

    def __init__(self):
        super().__init__(
            name='time_limit',
            description='The duration in seconds after which the induction of rules should be canceled. Must be at '
            + 'least 1 or 0, if no time limit should be set',
            mixin=TimeStoppingCriterionMixin)

    @override
    def _configure(self, config, value):
        if value == 0 and issubclass(type(config), NoTimeStoppingCriterionMixin):
            config.use_no_time_stopping_criterion()
        else:
            config.use_time_stopping_criterion().set_time_limit(value)


class PostOptimizationParameter(NominalParameter):
    """
    A parameter that allows to configure the method that should be used for post-optimization of a previously learned
    model.
    """

    POST_OPTIMIZATION_SEQUENTIAL = 'sequential'

    OPTION_NUM_ITERATIONS = 'num_iterations'

    OPTION_REFINE_HEADS = 'refine_heads'

    def __init__(self):
        super().__init__(name='post_optimization',
                         description='The method that should be used for post-optimization of a previous learned model')
        self.add_value(name=NONE, mixin=NoSequentialPostOptimizationMixin)
        self.add_value(name=self.POST_OPTIMIZATION_SEQUENTIAL,
                       mixin=SequentialPostOptimizationMixin,
                       options={self.OPTION_NUM_ITERATIONS, self.OPTION_REFINE_HEADS, OPTION_RESAMPLE_FEATURES})

    @override
    def _configure(self, config, value: str, options: Options):
        if value == NONE:
            config.use_no_sequential_post_optimization()
        elif value == self.POST_OPTIMIZATION_SEQUENTIAL:
            conf = config.use_sequential_post_optimization()
            conf.set_num_iterations(options.get_int(self.OPTION_NUM_ITERATIONS, conf.get_num_iterations()))
            conf.set_refine_heads(options.get_bool(self.OPTION_REFINE_HEADS, conf.are_heads_refined()))
            conf.set_resample_features(options.get_bool(OPTION_RESAMPLE_FEATURES, conf.are_features_resampled()))


RULE_LEARNER_PARAMETERS = {
    RandomStateParameter(),
    RuleInductionParameter(),
    FeatureBinningParameter(),
    OutputSamplingParameter(),
    InstanceSamplingParameter(),
    FeatureSamplingParameter(),
    PartitionSamplingParameter(),
    GlobalPruningParameter(),
    RulePruningParameter(),
    ParallelRuleRefinementParameter(),
    ParallelStatisticUpdateParameter(),
    ParallelPredictionParameter(),
    SizeStoppingCriterionParameter(),
    TimeStoppingCriterionParameter(),
    PostOptimizationParameter()
}
