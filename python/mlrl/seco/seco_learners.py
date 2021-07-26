#!/usr/bin/python

import logging as log

from sklearn.base import ClassifierMixin

from mlrl.common.cython.head_refinement import HeadRefinementFactory, SingleLabelHeadRefinementFactory, \
    FullHeadRefinementFactory
from mlrl.common.cython.input import LabelMatrix
from mlrl.common.cython.model import ModelBuilder
from mlrl.common.cython.output import Predictor
from mlrl.common.cython.post_processing import NoPostProcessor
from mlrl.common.cython.pruning import Pruning, IREP
from mlrl.common.cython.rule_induction import TopDownRuleInduction, SequentialRuleModelInduction
from mlrl.common.cython.statistics import StatisticsProviderFactory
from mlrl.common.rule_learners import HEAD_REFINEMENT_SINGLE
from mlrl.common.rule_learners import MLRuleLearner, SparsePolicy
from mlrl.common.rule_learners import create_pruning, create_feature_sub_sampling_factory, \
    create_instance_sub_sampling_factory, create_label_sub_sampling_factory, create_partition_sampling_factory, \
    create_max_conditions, create_stopping_criteria, create_min_coverage, create_max_head_refinements, \
    get_preferred_num_threads, parse_prefix_and_dict, get_int_argument, get_float_argument, create_thresholds_factory
from mlrl.seco.cython.head_refinement import PartialHeadRefinementFactory, LiftFunction, PeakLiftFunction
from mlrl.seco.cython.heuristics import Heuristic, Precision, Recall, Laplace, WRA, HammingLoss, FMeasure, MEstimate, \
    IREP, Ripper
from mlrl.seco.cython.model import DecisionListBuilder
from mlrl.seco.cython.output import LabelWiseClassificationPredictor
from mlrl.seco.cython.pruning import SecoPruning
from mlrl.seco.cython.rule_evaluation_label_wise import HeuristicLabelWiseRuleEvaluationFactory
from mlrl.seco.cython.statistics_label_wise import DenseLabelWiseStatisticsProviderFactory
from mlrl.seco.cython.stopping import CoverageStoppingCriterion

HEAD_REFINEMENT_PARTIAL = 'partial'

AVERAGING_LABEL_WISE = 'label-wise-averaging'

HEURISTIC_PRECISION = 'precision'

HEURISTIC_LAPLACE = 'laplace'

HEURISTIC_HAMMING_LOSS = 'hamming-loss'

HEURISTIC_RECALL = 'recall'

HEURISTIC_WRA = 'weighted-relative-accuracy'

HEURISTIC_F_MEASURE = 'f-measure'

HEURISTIC_M_ESTIMATE = 'm-estimate'

HEURISTIC_IREP = 'irep'

HEURISTIC_RIPPER = 'ripper'

LIFT_FUNCTION_PEAK = 'peak'

ARGUMENT_PEAK_LABEL = 'peak_label'

ARGUMENT_MAX_LIFT = 'max_lift'

ARGUMENT_CURVATURE = 'curvature'

ARGUMENT_BETA = 'beta'

ARGUMENT_M = 'm'


class SeparateAndConquerRuleLearner(MLRuleLearner, ClassifierMixin):
    """
    A scikit-multilearn implementation of an Separate-and-Conquer algorithm for learning multi-label classification
    rules.
    """

    def __init__(self, random_state: int = 1, feature_format: str = SparsePolicy.AUTO.value,
                 label_format: str = SparsePolicy.AUTO.value, max_rules: int = 500, time_limit: int = -1,
                 head_refinement: str = None, lift_function: str = LIFT_FUNCTION_PEAK, loss: str = AVERAGING_LABEL_WISE,
                 heuristic: str = HEURISTIC_PRECISION, pruning_heuristic: str = None, label_sub_sampling: str = None,
                 instance_sub_sampling: str = None, feature_sub_sampling: str = None, holdout: str = None,
                 feature_binning: str = None, pruning: str = None, prune_head: bool = False, min_coverage: int = 1,
                 max_conditions: int = -1, max_head_refinements: int = 1, num_threads_refinement: int = 1,
                 num_threads_update: int = 1, num_threads_prediction: int = 1, debugging_: str = None):
        """
        :param max_rules:                           The maximum number of rules to be induced (including the default
                                                    rule)
        :param time_limit:                          The duration in seconds after which the induction of rules should be
                                                    canceled
        :param head_refinement:                     The strategy that is used to find the heads of rules. Must be
                                                    `single-label`, `partial` or None, if the default strategy should be
                                                    used
        :param lift_function:                       The lift function to use. Must be `peak`. Additional arguments may
                                                    be provided as a dictionary, e.g.
                                                    `peak{\"peak_label\":10,\"max_lift\":2.0,\"curvature\":1.0}`
        :param loss:                                The loss function to be minimized. Must be `label-wise-averaging`
        :param heuristic:                           The heuristic to be minimized. Must be `precision`, `hamming-loss`,
                                                    `recall`, `weighted-relative-accuracy`, `f-measure`, `m-estimate`
                                                    or `laplace`. Additional arguments may be provided as a dictionary,
                                                    e.g. `f-measure{\"beta\":1.0}`
        :param pruning_heuristic                    The heuristic to be used for pruning. Can be any of the above as
                                                    well as `irep` and `ripper`.
        :param label_sub_sampling:                  The strategy that is used for sub-sampling the labels each time a
                                                    new classification rule is learned. Must be 'random-label-selection'
                                                    or None, if no sub-sampling should be used. Additional arguments may
                                                    be provided as a dictionary, e.g.
                                                    `random-label-selection{\"num_samples\":5}`
        :param instance_sub_sampling:               The strategy that is used for sub-sampling the training examples
                                                    each time a new classification rule is learned. Must be `bagging`,
                                                    `random-instance-selection` or None, if no sub-sampling should be
                                                    used. Additional arguments may be provided as a dictionary, e.g.
                                                    `bagging{\"sample_size\":0.5}`
        :param feature_sub_sampling:                The strategy that is used for sub-sampling the features each time a
                                                    classification rule is refined. Must be `random-feature-selection`
                                                    or None, if no sub-sampling should be used. Additional argument may
                                                    be provided as a dictionary, e.g.
                                                    `random-feature-selection{\"sample_size\":0.5}`
        :param holdout:                             The name of the strategy to be used for creating a holdout set. Must
                                                    be `random` or None, if no holdout set should be used. Additional
                                                    arguments may be provided as a dictionary, e.g.
                                                    `random{\"holdout_set_size\":0.5}`
        :param feature_binning:                     The strategy that is used for assigning examples to bins based on
                                                    their feature values. Must be `equal-width`, `equal-frequency` or
                                                    None, if no feature binning should be used. Additional arguments may
                                                    be provided as a dictionary, e.g. `equal-width{\"bin_ratio\":0.5}`
        :param pruning:                             The strategy that is used for pruning rules. Must be `irep` or None,
                                                    if no pruning should be used
        :param prune_head:                          If the head should be adapted when pruning
        :param min_coverage:                        The minimum number of training examples that must be covered by a
                                                    rule. Must be at least 1
        :param max_conditions:                      The maximum number of conditions to be included in a rule's body.
                                                    Must be at least 1 or -1, if the number of conditions should not be
                                                    restricted
        :param max_head_refinements:                The maximum number of times the head of a rule may be refined after
                                                    a new condition has been added to its body. Must be at least 1 or
                                                    -1, if the number of refinements should not be restricted
        :param num_threads_refinement:              The number of threads to be used to search for potential refinements
                                                    of rules or -1, if the number of cores that are available on the
                                                    machine should be used
        :param num_threads_update:                  The number of threads to be used to update statistics or -1, if the
                                                    number of cores that are available on the machine should be used
        :param num_threads_prediction:              The number of threads to be used to make predictions or -1, if the
                                                    number of cores that are available on the machine should be used
        :debugging:                                 What kind of debugging text should be printed during execution. Is
                                                    either 'full' or None, if no debugging text should be printed.
        """
        super().__init__(random_state, feature_format, label_format)
        self.max_rules = max_rules
        self.time_limit = time_limit
        self.head_refinement = head_refinement
        self.lift_function = lift_function
        self.loss = loss
        self.heuristic = heuristic
        self.pruning_heuristic = pruning_heuristic
        self.label_sub_sampling = label_sub_sampling
        self.instance_sub_sampling = instance_sub_sampling
        self.feature_sub_sampling = feature_sub_sampling
        self.holdout = holdout
        self.feature_binning = feature_binning
        self.pruning = pruning
        self.prune_head = prune_head
        self.min_coverage = min_coverage
        self.max_conditions = max_conditions
        self.max_head_refinements = max_head_refinements
        self.num_threads_refinement = num_threads_refinement
        self.num_threads_update = num_threads_update
        self.num_threads_prediction = num_threads_prediction
        self.debugging_ = debugging_

    def get_name(self) -> str:
        name = 'max-rules=' + str(self.max_rules)
        if self.head_refinement is not None:
            name += '_head-refinement=' + str(self.head_refinement)
        name += '_lift-function=' + str(self.lift_function)
        name += '_loss=' + str(self.loss)
        name += '_heuristic=' + str(self.heuristic)
        if self.pruning_heuristic is not None:
            name += '_pruning-heuristic=' + str(self.pruning_heuristic)
        if self.label_sub_sampling is not None:
            name += '_label-sub-sampling=' + str(self.label_sub_sampling)
        if self.instance_sub_sampling is not None:
            name += '_instance-sub-sampling=' + str(self.instance_sub_sampling)
        if self.feature_sub_sampling is not None:
            name += '_feature-sub-sampling=' + str(self.feature_sub_sampling)
        if self.holdout is not None:
            name += '_holdout=' + str(self.holdout)
        if self.feature_binning is not None:
            name += '_feature-binning=' + str(self.feature_binning)
        if self.pruning is not None:
            name += '_pruning=' + str(self.pruning)
        if self.prune_head:
            name += '_prune-head'
        if int(self.min_coverage) > 1:
            name += '_min-coverage=' + str(self.min_coverage)
        if int(self.max_conditions) != -1:
            name += '_max-conditions=' + str(self.max_conditions)
        if int(self.max_head_refinements) != -1:
            name += '_max-head-refinements=' + str(self.max_head_refinements)
        if int(self.random_state) != 1:
            name += '_random_state=' + str(self.random_state)
        if self.debugging_ is not None:
            name += '_debugging=' + str(self.debugging_)
        return name

    def _create_model_builder(self) -> ModelBuilder:
        return DecisionListBuilder()

    def _create_rule_model_induction(self, num_labels: int) -> SequentialRuleModelInduction:
        heuristic, pruning_heuristic = self.__create_heuristic()
        statistics_provider_factory = self.__create_statistics_provider_factory(heuristic, pruning_heuristic)
        num_threads_update = get_preferred_num_threads(self.num_threads_update)
        thresholds_factory = create_thresholds_factory(self.feature_binning, num_threads_update)
        min_coverage = create_min_coverage(self.min_coverage)
        max_conditions = create_max_conditions(self.max_conditions)
        max_head_refinements = create_max_head_refinements(self.max_head_refinements)
        num_threads_refinement = get_preferred_num_threads(self.num_threads_refinement)
        rule_induction = TopDownRuleInduction(min_coverage, max_conditions, max_head_refinements, False,
                                              num_threads_refinement)
        lift_function = self.__create_lift_function(num_labels)
        default_rule_head_refinement_factory = FullHeadRefinementFactory()
        head_refinement_factory = self.__create_head_refinement_factory(lift_function)
        label_sub_sampling_factory = create_label_sub_sampling_factory(self.label_sub_sampling, num_labels)
        instance_sub_sampling_factory = create_instance_sub_sampling_factory(self.instance_sub_sampling)
        feature_sub_sampling_factory = create_feature_sub_sampling_factory(self.feature_sub_sampling)
        partition_sampling_factory = create_partition_sampling_factory(self.holdout)
        pruning = self.__create_pruning(lift_function)
        post_processor = NoPostProcessor()
        stopping_criteria = create_stopping_criteria(int(self.max_rules), int(self.time_limit))
        stopping_criteria.append(CoverageStoppingCriterion(0))
        return SequentialRuleModelInduction(statistics_provider_factory, thresholds_factory, rule_induction,
                                            default_rule_head_refinement_factory, head_refinement_factory,
                                            label_sub_sampling_factory, instance_sub_sampling_factory,
                                            feature_sub_sampling_factory, partition_sampling_factory, pruning,
                                            post_processor, stopping_criteria)

    def __create_heuristic(self) -> (Heuristic, Heuristic):

        pruning_heuristic = self.pruning_heuristic
        if pruning_heuristic is not None:
            if pruning_heuristic == HEURISTIC_PRECISION:
                return_pruning_heuristic = Precision()
            elif pruning_heuristic == HEURISTIC_IREP:
                return_pruning_heuristic = IREP()
            elif pruning_heuristic == HEURISTIC_RIPPER:
                return_pruning_heuristic = Ripper()
            else:
                raise ValueError('Invalid value given for parameter \'pruning-heuristic\': ' + str(pruning_heuristic))
        else:
            return_pruning_heuristic = None

        heuristic = self.heuristic
        prefix, args = parse_prefix_and_dict(heuristic, [HEURISTIC_PRECISION, HEURISTIC_HAMMING_LOSS, HEURISTIC_RECALL,
                                                         HEURISTIC_LAPLACE, HEURISTIC_WRA, HEURISTIC_F_MEASURE,
                                                         HEURISTIC_M_ESTIMATE])

        if prefix == HEURISTIC_PRECISION:
            return_heuristic = Precision()
        elif prefix == HEURISTIC_HAMMING_LOSS:
            return_heuristic = HammingLoss()
        elif prefix == HEURISTIC_RECALL:
            return_heuristic = Recall()
        elif prefix == HEURISTIC_LAPLACE:
            return_heuristic = Laplace()
        elif prefix == HEURISTIC_WRA:
            return_heuristic = WRA()
        elif prefix == HEURISTIC_F_MEASURE:
            beta = get_float_argument(args, ARGUMENT_BETA, 0.5, lambda x: x >= 0)
            return_heuristic = FMeasure(beta)
        elif prefix == HEURISTIC_M_ESTIMATE:
            m = get_float_argument(args, ARGUMENT_M, 22.466, lambda x: x >= 0)
            return_heuristic = MEstimate(m)
        else:
            raise ValueError('Invalid value given for parameter \'heuristic\': ' + str(heuristic))

        if return_pruning_heuristic is None:
            return return_heuristic, return_heuristic
        else:
            return return_heuristic, return_pruning_heuristic

    def __create_statistics_provider_factory(self, heuristic: Heuristic, pruning_heuristic: Heuristic) \
            -> StatisticsProviderFactory:
        loss = self.loss

        if loss == AVERAGING_LABEL_WISE:
            default_rule_evaluation_factory = HeuristicLabelWiseRuleEvaluationFactory(heuristic, pruning_heuristic,
                                                                                      predictMajority=True)
            rule_evaluation_factory = HeuristicLabelWiseRuleEvaluationFactory(heuristic, pruning_heuristic)
            return DenseLabelWiseStatisticsProviderFactory(default_rule_evaluation_factory, rule_evaluation_factory)
        raise ValueError('Invalid value given for parameter \'loss\': ' + str(loss))

    def __create_lift_function(self, num_labels: int) -> LiftFunction:
        lift_function = self.lift_function

        prefix, args = parse_prefix_and_dict(lift_function, [LIFT_FUNCTION_PEAK])

        if prefix == LIFT_FUNCTION_PEAK:
            peak_label = get_int_argument(args, ARGUMENT_PEAK_LABEL, int(num_labels / 2) + 1,
                                          lambda x: 1 <= x <= num_labels)
            max_lift = get_float_argument(args, ARGUMENT_MAX_LIFT, 1.5, lambda x: x >= 1)
            curvature = get_float_argument(args, ARGUMENT_CURVATURE, 1.0, lambda x: x > 0)
            return PeakLiftFunction(num_labels, peak_label, max_lift, curvature)

        raise ValueError('Invalid value given for parameter \'lift_function\': ' + str(lift_function))

    def __create_head_refinement_factory(self, lift_function: LiftFunction) -> HeadRefinementFactory:
        head_refinement = self.head_refinement

        if head_refinement is None:
            return SingleLabelHeadRefinementFactory()
        elif head_refinement == HEAD_REFINEMENT_SINGLE:
            return SingleLabelHeadRefinementFactory()
        elif head_refinement == HEAD_REFINEMENT_PARTIAL:
            return PartialHeadRefinementFactory(lift_function)
        raise ValueError('Invalid value given for parameter \'head_refinement\': ' + str(head_refinement))

    def _create_predictor(self, num_labels: int, label_matrix: LabelMatrix) -> Predictor:
        return self.__create_label_wise_predictor(num_labels)

    def __create_label_wise_predictor(self, num_labels: int) -> LabelWiseClassificationPredictor:
        num_threads = get_preferred_num_threads(self.num_threads_prediction)
        return LabelWiseClassificationPredictor(num_labels, num_threads)

    def __create_pruning(self, lift_function: LiftFunction) -> Pruning:
        if self.prune_head and self.pruning is not None:
            if self.head_refinement != HEAD_REFINEMENT_SINGLE:
                return SecoPruning(lift_function)
            else:
                log.warning('Parameter \'prune_head\' does not have any effect, because '
                            + 'parameter \'head_refinement\' is set to single')
        return create_pruning(self.pruning, self.instance_sub_sampling)
