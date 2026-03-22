"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides a scikit-learn implementation of a Separate-and-Conquer (SeCo) algorithm for learning multi-label
classification rules.
"""

from typing import Any, override

from mlrl.common.learners import ClassificationRuleLearner, configure_rule_learner

from mlrl.seco.config.parameters import SECO_CLASSIFIER_PARAMETERS
from mlrl.seco.cython.learner_seco import SeCoClassifier as SeCoWrapper, SeCoClassifierConfig


class SeCoClassifier(ClassificationRuleLearner):
    """
    A scikit-learn implementation of a Separate-and-Conquer (SeCo) algorithm for learning multi-label classification
    rules.
    """

    def __init__(
        self,
        random_state: int | None = None,
        feature_format: str | None = None,
        output_format: str | None = None,
        prediction_format: str | None = None,
        rule_induction: str | None = None,
        max_rules: int | None = None,
        time_limit: int | None = None,
        post_optimization: str | None = None,
        head_type: str | None = None,
        lift_function: str | None = None,
        heuristic: str | None = None,
        pruning_heuristic: str | None = None,
        output_sampling: str | None = None,
        instance_sampling: str | None = None,
        feature_sampling: str | None = None,
        holdout: str | None = None,
        feature_binning: str | None = None,
        rule_pruning: str | None = None,
        parallel_rule_refinement: str | None = None,
        parallel_statistic_update: str | None = None,
        parallel_prediction: str | None = None,
    ):
        """
        :param random_state:                The seed to be used by RNGs. Must be at least 0
        :param rule_induction:              An algorithm to be used for the induction of individual rules. Must be
                                            'top-down-greedy' or 'top-down-beam-search'. For additional options refer to
                                            the documentation
        :param max_rules:                   The maximum number of rules to be learned (including the default rule). Must
                                            be at least 1 or 0, if the number of rules should not be restricted
        :param time_limit:                  The duration in seconds after which the induction of rules should be
                                            canceled. Must be at least 1 or 0, if no time limit should be set
        :param post_optimization:           Whether each rule in a previously learned model should be optimized by being
                                            relearned in the context of the other rules or not. Must be 'true' or
                                            'false'. For additional options refer to the documentation
        :param head_type:                   The type of the rule heads that should be used. Must be 'single' or
                                            'partial'
        :param lift_function:               The lift function that should be used for the induction of partial rule
                                            heads. Must be 'peak', 'kln' or 'none'. For additional options refer to the
                                            documentation
        :param heuristic:                   The heuristic to be optimized. Must be 'accuracy', 'precision', 'recall',
                                            'weighted-relative-accuracy', 'f-measure', 'm-estimate' or 'laplace'. For
                                            additional options refer to the documentation
        :param pruning_heuristic:           The heuristic to be optimized when pruning rules. Must be 'accuracy',
                                            'precision', 'recall', 'weighted-relative-accuracy', 'f-measure',
                                            'm-estimate' or 'laplace'. For additional options refer to the documentation
        :param output_sampling:             The strategy that should be used to sample from the available outputs
                                            whenever a new rule is learned. Must be 'round-robin', 'without-replacement'
                                            or 'none', if no sampling should be used. For additional options refer to
                                            the documentation
        :param instance_sampling:           The strategy that should be used to sample from the available the training
                                            examples whenever a new rule is learned. Must be 'with-replacement',
                                            'without-replacement', 'stratified-output-wise', 'stratified-example-wise'
                                            or 'none', if no sampling should be used. For additional options refer to
                                            the documentation
        :param feature_sampling:            The strategy that is used to sample from the available features whenever a
                                            rule is refined. Must be 'without-replacement' or 'none', if no sampling
                                            should be used. For additional options refer to the documentation
        :param holdout:                     The name of the strategy that should be used to creating a holdout set. Must
                                            be 'random', 'stratified-output-wise', 'stratified-example-wise' or 'none',
                                            if no holdout set should be used. For additional options refer to the
                                            documentation
        :param feature_binning:             The strategy that should be used to assign examples to bins based on their
                                            feature values. Must be 'equal-width', 'equal-frequency' or 'none', if no
                                            feature binning should be used. For additional options refer to the
                                            documentation
        :param rule_pruning:                The strategy that should be used to prune individual rules. Must be 'irep'
                                            or 'none', if no pruning should be used
        :param parallel_rule_refinement:    Whether potential refinements of rules should be searched for in parallel or
                                            not. Must be 'true', 'false' or 'auto', if the most suitable strategy should
                                            be chosen automatically depending on the loss function. For additional
                                            options refer to the documentation
        :param parallel_statistic_update:   Whether the confusion matrices for different examples should be updated in
                                            parallel or not. Must be 'true', 'false' or 'auto', if the most suitable
                                            strategy should be chosen automatically, depending on the loss function. For
                                            additional options refer to the documentation
        :param parallel_prediction:         Whether predictions for different examples should be obtained in parallel or
                                            not. Must be 'true' or 'false'. For additional options refer to the
                                            documentation
        """
        super().__init__(feature_format, output_format, prediction_format)
        self.random_state = random_state
        self.rule_induction = rule_induction
        self.max_rules = max_rules
        self.time_limit = time_limit
        self.post_optimization = post_optimization
        self.head_type = head_type
        self.lift_function = lift_function
        self.heuristic = heuristic
        self.pruning_heuristic = pruning_heuristic
        self.output_sampling = output_sampling
        self.instance_sampling = instance_sampling
        self.feature_sampling = feature_sampling
        self.holdout = holdout
        self.feature_binning = feature_binning
        self.rule_pruning = rule_pruning
        self.parallel_rule_refinement = parallel_rule_refinement
        self.parallel_statistic_update = parallel_statistic_update
        self.parallel_prediction = parallel_prediction

    @override
    def _create_learner(self) -> Any:
        config = SeCoClassifierConfig()
        configure_rule_learner(self, config, SECO_CLASSIFIER_PARAMETERS)
        return SeCoWrapper(config)
