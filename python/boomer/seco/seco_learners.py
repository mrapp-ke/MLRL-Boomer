#!/usr/bin/python

from boomer.common.head_refinement import SingleLabelHeadRefinement, HeadRefinement
from boomer.common.prediction import Predictor, DensePredictor, SignFunction
from boomer.common.rule_induction import ExactGreedyRuleInduction
from boomer.common.rules import ModelBuilder, RuleListBuilder
from boomer.common.sequential_rule_induction import SequentialRuleInduction
from boomer.seco.coverage_losses import CoverageLoss
from boomer.seco.head_refinement import PartialHeadRefinement
from boomer.seco.heuristics import Heuristic, HammingLoss, Precision, Recall, WeightedRelativeAccuracy, FMeasure, \
    MEstimate
from boomer.seco.label_wise_averaging import LabelWiseAveraging
from boomer.seco.lift_functions import LiftFunction, PeakLiftFunction
from boomer.seco.stopping_criteria import UncoveredLabelsCriterion

from boomer.common.rule_learners import HEAD_REFINEMENT_SINGLE
from boomer.common.rule_learners import MLRuleLearner
from boomer.common.rule_learners import create_pruning, create_feature_sub_sampling, create_instance_sub_sampling, \
    create_label_sub_sampling, create_max_conditions, create_stopping_criteria, create_min_coverage, \
    create_max_head_refinements, parse_prefix_and_dict, get_int_argument, get_float_argument

HEAD_REFINEMENT_PARTIAL = 'partial'

AVERAGING_LABEL_WISE = 'label-wise-averaging'

HEURISTIC_PRECISION = 'precision'

HEURISTIC_HAMMING_LOSS = 'hamming-loss'

HEURISTIC_RECALL = 'recall'

HEURISTIC_WRA = 'weighted-relative-accuracy'

HEURISTIC_F_MEASURE = 'f-measure'

HEURISTIC_M_ESTIMATE = 'm-estimate'

LIFT_FUNCTION_PEAK = 'peak'

ARGUMENT_PEAK_LABEL = 'peak_label'

ARGUMENT_MAX_LIFT = 'max_lift'

ARGUMENT_CURVATURE = 'curvature'

ARGUMENT_BETA = 'beta'

ARGUMENT_M = 'm'


class SeparateAndConquerRuleLearner(MLRuleLearner):
    """
    A scikit-multilearn implementation of an Separate-and-Conquer algorithm for learning multi-label classification
    rules.
    """

    def __init__(self, random_state: int = 1, max_rules: int = 500, time_limit: int = -1, head_refinement: str = None,
                 lift_function: str = LIFT_FUNCTION_PEAK, loss: str = AVERAGING_LABEL_WISE,
                 heuristic: str = HEURISTIC_PRECISION, label_sub_sampling: str = None,
                 instance_sub_sampling: str = None, feature_sub_sampling: str = None, pruning: str = None,
                 min_coverage: int = 1, max_conditions: int = -1, max_head_refinements: int = 1):
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
                                                    `recall`, `weighted-relative-accuracy`, `f-measure` or `m-estimate`.
                                                    Additional arguments may be provided as a dictionary, e.g.
                                                    `f-measure{\"beta\":1.0}`
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
        :param pruning:                             The strategy that is used for pruning rules. Must be `irep` or None,
                                                    if no pruning should be used
        :param min_coverage:                        The minimum number of training examples that must be covered by a
                                                    rule. Must be at least 1
        :param max_conditions:                      The maximum number of conditions to be included in a rule's body.
                                                    Must be at least 1 or -1, if the number of conditions should not be
                                                    restricted
        :param max_head_refinements:                The maximum number of times the head of a rule may be refined after
                                                    a new condition has been added to its body. Must be at least 1 or
                                                    -1, if the number of refinements should not be restricted
        """
        super().__init__(random_state)
        self.max_rules = max_rules
        self.time_limit = time_limit
        self.head_refinement = head_refinement
        self.lift_function = lift_function
        self.loss = loss
        self.heuristic = heuristic
        self.label_sub_sampling = label_sub_sampling
        self.instance_sub_sampling = instance_sub_sampling
        self.feature_sub_sampling = feature_sub_sampling
        self.pruning = pruning
        self.min_coverage = min_coverage
        self.max_conditions = max_conditions
        self.max_head_refinements = max_head_refinements

    def get_name(self) -> str:
        name = 'max-rules=' + str(self.max_rules)
        if self.head_refinement is not None:
            name += '_head-refinement=' + str(self.head_refinement)
        name += '_lift-function=' + str(self.lift_function)
        name += '_loss=' + str(self.loss)
        name += '_heuristic=' + str(self.heuristic)
        if self.label_sub_sampling is not None:
            name += '_label-sub-sampling=' + str(self.label_sub_sampling)
        if self.instance_sub_sampling is not None:
            name += '_instance-sub-sampling=' + str(self.instance_sub_sampling)
        if self.feature_sub_sampling is not None:
            name += '_feature-sub-sampling=' + str(self.feature_sub_sampling)
        if self.pruning is not None:
            name += '_pruning=' + str(self.pruning)
        if int(self.min_coverage) > 1:
            name += '_min-coverage=' + str(self.min_coverage)
        if int(self.max_conditions) != -1:
            name += '_max-conditions=' + str(self.max_conditions)
        if int(self.max_head_refinements) != 1:
            name += '_max-head-refinements=' + str(self.max_head_refinements)
        if int(self.random_state) != 1:
            name += '_random_state=' + str(self.random_state)
        return name

    def _create_model_builder(self) -> ModelBuilder:
        return RuleListBuilder(use_mask=True, default_rule_at_end=True)

    def _create_sequential_rule_induction(self, num_labels: int) -> SequentialRuleInduction:
        rule_induction = ExactGreedyRuleInduction()
        heuristic = self.__create_heuristic()
        loss = self.__create_loss(heuristic)
        lift_function = self.__create_lift_function(num_labels)
        head_refinement = self.__create_head_refinement(lift_function)
        label_sub_sampling = create_label_sub_sampling(self.label_sub_sampling, num_labels)
        instance_sub_sampling = create_instance_sub_sampling(self.instance_sub_sampling)
        feature_sub_sampling = create_feature_sub_sampling(self.feature_sub_sampling)
        pruning = create_pruning(self.pruning)
        min_coverage = create_min_coverage(self.min_coverage)
        max_conditions = create_max_conditions(self.max_conditions)
        max_head_refinements = create_max_head_refinements(self.max_head_refinements)
        stopping_criteria = create_stopping_criteria(int(self.max_rules), int(self.time_limit))
        stopping_criteria.append(UncoveredLabelsCriterion(loss, 0))
        return SequentialRuleInduction(rule_induction, head_refinement, loss, stopping_criteria, label_sub_sampling,
                                       instance_sub_sampling, feature_sub_sampling, pruning, None, min_coverage,
                                       max_conditions, max_head_refinements)

    def __create_heuristic(self) -> Heuristic:
        heuristic = self.heuristic
        prefix, args = parse_prefix_and_dict(heuristic, [HEURISTIC_PRECISION, HEURISTIC_HAMMING_LOSS, HEURISTIC_RECALL,
                                                         HEURISTIC_WRA, HEURISTIC_F_MEASURE, HEURISTIC_M_ESTIMATE])

        if prefix == HEURISTIC_PRECISION:
            return Precision()
        elif prefix == HEURISTIC_HAMMING_LOSS:
            return HammingLoss()
        elif prefix == HEURISTIC_RECALL:
            return Recall()
        elif prefix == HEURISTIC_WRA:
            return WeightedRelativeAccuracy()
        elif prefix == HEURISTIC_F_MEASURE:
            beta = get_float_argument(args, ARGUMENT_BETA, 0.5, lambda x: x >= 0)
            return FMeasure(beta)
        elif prefix == HEURISTIC_M_ESTIMATE:
            m = get_float_argument(args, ARGUMENT_M, 22.466, lambda x: x >= 0)
            return MEstimate(m)
        raise ValueError('Invalid value given for parameter \'heuristic\': ' + str(heuristic))

    def __create_loss(self, heuristic: Heuristic) -> CoverageLoss:
        loss = self.loss

        if loss == AVERAGING_LABEL_WISE:
            return LabelWiseAveraging(heuristic)
        raise ValueError('Invalid value given for parameter \'loss\': ' + str(loss))

    def __create_lift_function(self, num_labels: int) -> LiftFunction:
        lift_function = self.lift_function

        prefix, args = parse_prefix_and_dict(lift_function, [LIFT_FUNCTION_PEAK])

        if prefix == LIFT_FUNCTION_PEAK:
            peak_label = get_int_argument(args, ARGUMENT_PEAK_LABEL, int(num_labels / 2) + 1,
                                          lambda x: 1 <= x <= num_labels)
            max_lift = get_float_argument(args, ARGUMENT_MAX_LIFT, 2.0, lambda x: x >= 1)
            curvature = get_float_argument(args, ARGUMENT_CURVATURE, 1.0, lambda x: x > 0)
            return PeakLiftFunction(num_labels, peak_label, max_lift, curvature)

        raise ValueError('Invalid value given for parameter \'lift_function\': ' + str(lift_function))

    def __create_head_refinement(self, lift_function: LiftFunction) -> HeadRefinement:
        head_refinement = self.head_refinement

        if head_refinement is None:
            return SingleLabelHeadRefinement()
        elif head_refinement == HEAD_REFINEMENT_SINGLE:
            return SingleLabelHeadRefinement()
        elif head_refinement == HEAD_REFINEMENT_PARTIAL:
            return PartialHeadRefinement(lift_function)
        raise ValueError('Invalid value given for parameter \'head_refinement\': ' + str(head_refinement))

    def _create_predictor(self) -> Predictor:
        return DensePredictor(SignFunction())
