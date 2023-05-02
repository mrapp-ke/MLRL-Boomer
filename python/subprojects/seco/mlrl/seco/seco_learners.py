"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides a scikit-learn implementation of a Separate-and-Conquer (SeCo) algorithm for learning multi-label
classification rules.
"""
from mlrl.common.config import configure_rule_induction, configure_label_sampling, configure_instance_sampling, \
    configure_feature_sampling, configure_partition_sampling, configure_rule_pruning, \
    configure_parallel_rule_refinement, configure_parallel_statistic_update, configure_parallel_prediction, \
    configure_size_stopping_criterion, configure_time_stopping_criterion, configure_sequential_post_optimization
from mlrl.common.cython.learner import RuleLearner as RuleLearnerWrapper
from mlrl.common.options import parse_param_and_options
from mlrl.common.rule_learners import RuleLearner, SparsePolicy, get_string, get_int
from mlrl.seco.config import HEURISTIC_ACCURACY, HEURISTIC_PRECISION, HEURISTIC_RECALL, HEURISTIC_LAPLACE, \
    HEURISTIC_WRA, HEURISTIC_F_MEASURE, HEURISTIC_M_ESTIMATE, OPTION_M, OPTION_BETA
from mlrl.seco.config import configure_head_type, configure_lift_function, configure_accuracy_heuristic, \
    configure_precision_heuristic, configure_recall_heuristic, configure_laplace_heuristic, configure_wra_heuristic, \
    configure_f_measure_heuristic, configure_m_estimate_heuristic, configure_accuracy_pruning_heuristic, \
    configure_precision_pruning_heuristic, configure_recall_pruning_heuristic, configure_laplace_pruning_heuristic, \
    configure_wra_pruning_heuristic, configure_f_measure_pruning_heuristic, configure_m_estimate_pruning_heuristic
from mlrl.seco.cython.learner_seco import MultiLabelSeCoRuleLearner as MultiLabelSeCoRuleLearnerWrapper, \
    MultiLabelSeCoRuleLearnerConfig
from sklearn.base import ClassifierMixin, MultiOutputMixin
from typing import Dict, Set, Optional

HEURISTIC_VALUES: Dict[str, Set[str]] = {
    HEURISTIC_ACCURACY: {},
    HEURISTIC_PRECISION: {},
    HEURISTIC_RECALL: {},
    HEURISTIC_LAPLACE: {},
    HEURISTIC_WRA: {},
    HEURISTIC_F_MEASURE: {OPTION_BETA},
    HEURISTIC_M_ESTIMATE: {OPTION_M}
}


class MultiLabelSeCoRuleLearner(RuleLearner, ClassifierMixin, MultiOutputMixin):
    """
    A scikit-learn implementation of a Separate-and-Conquer (SeCo) algorithm for learning multi-label classification
    rules.
    """

    def __init__(self,
                 random_state: int = 1,
                 feature_format: str = SparsePolicy.AUTO.value,
                 label_format: str = SparsePolicy.AUTO.value,
                 prediction_format: str = SparsePolicy.AUTO.value,
                 rule_induction: Optional[str] = None,
                 max_rules: Optional[int] = None,
                 time_limit: Optional[int] = None,
                 sequential_post_optimization: Optional[str] = None,
                 head_type: Optional[str] = None,
                 lift_function: Optional[str] = None,
                 heuristic: Optional[str] = None,
                 pruning_heuristic: Optional[str] = None,
                 label_sampling: Optional[str] = None,
                 instance_sampling: Optional[str] = None,
                 feature_sampling: Optional[str] = None,
                 holdout: Optional[str] = None,
                 rule_pruning: Optional[str] = None,
                 parallel_rule_refinement: Optional[str] = None,
                 parallel_statistic_update: Optional[str] = None,
                 parallel_prediction: Optional[str] = None):
        """
        :param rule_induction:                  An algorithm to be used for the induction of individual rules. Must be
                                                'top-down-greedy' or 'top-down-beam-search'. For additional options
                                                refer to the documentation
        :param max_rules:                       The maximum number of rules to be learned (including the default rule).
                                                Must be at least 1 or 0, if the number of rules should not be restricted
        :param time_limit:                      The duration in seconds after which the induction of rules should be
                                                canceled. Must be at least 1 or 0, if no time limit should be set
        :param sequential_post_optimization:    Whether each rule in a previously learned model should be optimized by
                                                being relearned in the context of the other rules or not. Must be 'true'
                                                or 'false'. For additional options refer to the documentation
        :param head_type:                       The type of the rule heads that should be used. Must be 'single-label'
                                                or 'partial'
        :param lift_function:                   The lift function that should be used for the induction of partial rule
                                                heads. Must be 'peak', 'kln' or 'none'. For additional options refer to
                                                the documentation
        :param heuristic:                       The heuristic to be optimized. Must be 'accuracy', 'precision',
                                                'recall', 'weighted-relative-accuracy', 'f-measure', 'm-estimate' or
                                                'laplace'. For additional options refer to the documentation
        :param pruning_heuristic:               The heuristic to be optimized when pruning rules. Must be 'accuracy',
                                                'precision', 'recall', 'weighted-relative-accuracy', 'f-measure',
                                                'm-estimate' or 'laplace'. For additional options refer to the
                                                documentation
        :param label_sampling:                  The strategy that should be used to sample from the available labels
                                                whenever a new rule is learned. Must be 'without-replacement' or 'none',
                                                if no sampling should be used. For additional options refer to the
                                                documentation
        :param instance_sampling:               The strategy that should be used to sample from the available the
                                                training examples whenever a new rule is learned. Must be
                                                'with-replacement', 'without-replacement', 'stratified_label_wise',
                                                'stratified_example_wise' or 'none', if no sampling should be used. For
                                                additional options refer to the documentation
        :param feature_sampling:                The strategy that is used to sample from the available features whenever
                                                a rule is refined. Must be 'without-replacement' or 'none', if no
                                                sampling should be used. For additional options refer to the
                                                documentation
        :param holdout:                         The name of the strategy that should be used to creating a holdout set.
                                                Must be 'random', 'stratified-label-wise', 'stratified-example-wise' or
                                                'none', if no holdout set should be used. For additional options refer
                                                to the documentation
        :param rule_pruning:                    The strategy that should be used to prune individual rules. Must be
                                                'irep' or 'none', if no pruning should be used
        :param parallel_rule_refinement:        Whether potential refinements of rules should be searched for in
                                                parallel or not. Must be 'true', 'false' or 'auto', if the most suitable
                                                strategy should be chosen automatically depending on the loss function.
                                                For additional options refer to the documentation
        :param parallel_statistic_update:       Whether the confusion matrices for different examples should be updated
                                                in parallel or not. Must be 'true', 'false' or 'auto', if the most
                                                suitable strategy should be chosen automatically, depending on the loss
                                                function. For additional options refer to the documentation
        :param parallel_prediction:             Whether predictions for different examples should be obtained in
                                                parallel or not. Must be 'true' or 'false'. For additional options refer
                                                to the documentation
        """
        super().__init__(random_state, feature_format, label_format, prediction_format)
        self.rule_induction = rule_induction
        self.max_rules = max_rules
        self.time_limit = time_limit
        self.sequential_post_optimization = sequential_post_optimization
        self.head_type = head_type
        self.lift_function = lift_function
        self.heuristic = heuristic
        self.pruning_heuristic = pruning_heuristic
        self.label_sampling = label_sampling
        self.instance_sampling = instance_sampling
        self.feature_sampling = feature_sampling
        self.holdout = holdout
        self.rule_pruning = rule_pruning
        self.parallel_rule_refinement = parallel_rule_refinement
        self.parallel_statistic_update = parallel_statistic_update
        self.parallel_prediction = parallel_prediction

    def _create_learner(self) -> RuleLearnerWrapper:
        config = MultiLabelSeCoRuleLearnerConfig()
        configure_rule_induction(config, get_string(self.rule_induction))
        configure_label_sampling(config, get_string(self.label_sampling))
        configure_instance_sampling(config, get_string(self.instance_sampling))
        configure_feature_sampling(config, get_string(self.feature_sampling))
        configure_partition_sampling(config, get_string(self.holdout))
        configure_rule_pruning(config, get_string(self.rule_pruning))
        configure_parallel_rule_refinement(config, get_string(self.parallel_rule_refinement))
        configure_parallel_statistic_update(config, get_string(self.parallel_statistic_update))
        configure_parallel_prediction(config, get_string(self.parallel_prediction))
        configure_size_stopping_criterion(config, max_rules=get_int(self.max_rules))
        configure_time_stopping_criterion(config, time_limit=get_int(self.time_limit))
        configure_sequential_post_optimization(config, get_string(self.sequential_post_optimization))
        configure_head_type(config, get_string(self.head_type))
        configure_lift_function(config, get_string(self.lift_function))
        self.__configure_heuristic(config)
        self.__configure_pruning_heuristic(config)
        return MultiLabelSeCoRuleLearnerWrapper(config)

    def __configure_heuristic(self, config: MultiLabelSeCoRuleLearnerConfig):
        heuristic = get_string(self.heuristic)

        if heuristic is not None:
            value, options = parse_param_and_options('heuristic', heuristic, HEURISTIC_VALUES)
            configure_accuracy_heuristic(config, value)
            configure_precision_heuristic(config, value)
            configure_recall_heuristic(config, value)
            configure_laplace_heuristic(config, value)
            configure_wra_heuristic(config, value)
            configure_f_measure_heuristic(config, value, options)
            configure_m_estimate_heuristic(config, value, options)

    def __configure_pruning_heuristic(self, config: MultiLabelSeCoRuleLearnerConfig):
        pruning_heuristic = get_string(self.pruning_heuristic)

        if pruning_heuristic is not None:
            value, options = parse_param_and_options('pruning_heuristic', pruning_heuristic, HEURISTIC_VALUES)
            configure_accuracy_pruning_heuristic(config, value)
            configure_precision_pruning_heuristic(config, value)
            configure_recall_pruning_heuristic(config, value)
            configure_laplace_pruning_heuristic(config, value)
            configure_wra_pruning_heuristic(config, value)
            configure_f_measure_pruning_heuristic(config, value, options)
            configure_m_estimate_pruning_heuristic(config, value, options)
