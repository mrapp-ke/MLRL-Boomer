"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides scikit-learn implementations of boosting algorithms.
"""
from typing import Any, Optional

from mlrl.common.learners import ClassificationRuleLearner, RegressionRuleLearner, configure_rule_learner
from mlrl.common.mixins import ClassifierMixin, RegressorMixin

from mlrl.boosting.config.parameters import BOOMER_CLASSIFIER_PARAMETERS, BOOMER_REGRESSOR_PARAMETERS
from mlrl.boosting.cython.learner_boomer import BoomerClassifier as BoomerClassifierWrapper, BoomerClassifierConfig, \
    BoomerRegressor as BoomerRegressorWrapper, BoomerRegressorConfig


class BoomerClassifier(ClassificationRuleLearner, ClassifierMixin):
    """
    A scikit-learn implementation of "BOOMER", an algorithm for learning gradient boosted multi-output rules, that can
    be used in classification problems.
    """

    def __init__(self,
                 random_state: Optional[int] = None,
                 feature_format: Optional[str] = None,
                 output_format: Optional[str] = None,
                 prediction_format: Optional[str] = None,
                 statistic_format: Optional[str] = None,
                 statistic_type: Optional[str] = None,
                 default_rule: Optional[str] = None,
                 rule_induction: Optional[str] = None,
                 max_rules: Optional[int] = None,
                 time_limit: Optional[int] = None,
                 global_pruning: Optional[str] = None,
                 post_optimization: Optional[str] = None,
                 head_type: Optional[str] = None,
                 loss: Optional[str] = None,
                 marginal_probability_calibration: Optional[str] = None,
                 joint_probability_calibration: Optional[str] = None,
                 binary_predictor: Optional[str] = None,
                 probability_predictor: Optional[str] = None,
                 output_sampling: Optional[str] = None,
                 instance_sampling: Optional[str] = None,
                 feature_sampling: Optional[str] = None,
                 holdout: Optional[str] = None,
                 feature_binning: Optional[str] = None,
                 label_binning: Optional[str] = None,
                 rule_pruning: Optional[str] = None,
                 shrinkage: Optional[float] = 0.3,
                 l1_regularization_weight: Optional[float] = None,
                 l2_regularization_weight: Optional[float] = None,
                 parallel_rule_refinement: Optional[str] = None,
                 parallel_statistic_update: Optional[str] = None,
                 parallel_prediction: Optional[str] = None):
        """
        :param random_state:                        The seed to be used by RNGs. Must be at least 1
        :param statistic_format:                    The format to be used for representing gradients and Hessians. Must
                                                    be 'dense', 'sparse' or 'auto', if the most suitable format should
                                                    be chosen automatically
        :param statistic_type:                      The data type to be used for representing gradients and Hessians.
                                                    Must be '32-bit' or '64-bit'
        :param default_rule:                        Whether a default rule should be induced or not. Must be 'true',
                                                    'false' or 'auto', if it should be decided automatically whether a
                                                    default rule should be induced or not
        :param rule_induction:                      The algorithm that should be used for the induction of individual
                                                    rules. Must be 'top-down-greedy' or 'top-down-beam-search'. For
                                                    additional options refer to the documentation
        :param max_rules:                           The maximum number of rules to be learned (including the default
                                                    rule). Must be at least 1 or 0, if the number of rules should not be
                                                    restricted
        :param time_limit:                          The duration in seconds after which the induction of rules should be
                                                    canceled. Must be at least 1 or 0, if no time limit should be set
        :param global_pruning:                      The strategy that should be used for pruning entire rules. Must be
                                                    'pre-pruning', 'post-pruning' or 'none', if no pruning should be
                                                    used. For additional options refer to the documentation
        :param post_optimization:                   Whether each rule in a previously learned model should be optimized
                                                    by being relearned in the context of the other rules or not. Must be
                                                    'true' or 'false'. For additional options refer to the documentation
        :param head_type:                           The type of the rule heads that should be used. Must be 'single',
                                                    'complete', 'partial-fixed', 'partial-dynamic' or 'auto', if the
                                                    type of the heads should be chosen automatically. For additional
                                                    options refer to the documentation
        :param loss:                                The loss function to be minimized. Must be
                                                    'squared-error-decomposable', 'squared-error-non-decomposable',
                                                    'squared-hinge-decomposable', 'squared-hinge-non-decomposable',
                                                    'logistic-decomposable' or 'logistic-non-decomposable'
        :param marginal_probability_calibration:    The method that should be used for the calibration of marginal
                                                    probabilities. Must be 'isotonic' or 'none', if no probability
                                                    calibration should be used.
        :param joint_probability_calibration:       The method that should be used for the calibration of joint
                                                    probabilities. Must be 'isotonic' or 'none', if no probability
                                                    calibration should be used
        :param binary_predictor:                    The strategy that should be used for predicting binary labels. Must
                                                    be 'output-wise', 'example-wise', 'gfm' or 'auto', if the most
                                                    suitable strategy should be chosen automatically, depending on the
                                                    loss function
        :param probability_predictor:               The strategy that should be used for predicting probabilities. Must
                                                    be 'output-wise', 'marginalized' or 'auto', if the most suitable
                                                    strategy should be chosen automatically, depending on the loss
                                                    function
        :param output_sampling:                     The strategy that should be used to sample from the available
                                                    outputs whenever a new rule is learned. Must be 'round-robin',
                                                    'without-replacement' or 'none', if no sampling should be used. For
                                                    additional options refer to the documentation
        :param instance_sampling:                   The strategy that should be used to sample from the available the
                                                    training examples whenever a new rule is learned. Must be
                                                    'with-replacement', 'without-replacement', 'stratified-output-wise',
                                                    'stratified-example-wise' or 'none', if no sampling should be used.
                                                    For additional options refer to the documentation
        :param feature_sampling:                    The strategy that is used to sample from the available features
                                                    whenever a rule is refined. Must be 'without-replacement' or 'none',
                                                    if no sampling should be used. For additional options refer to the
                                                    documentation
        :param holdout:                             The name of the strategy that should be used to create a holdout
                                                    set. Must be 'random', 'stratified-output-wise',
                                                    'stratified-example-wise' or 'none', if no holdout set should be
                                                    used. If set to 'auto', the most suitable strategy is chosen
                                                    automatically depending on whether a holdout set is needed and
                                                    depending on the loss function. For additional options refer to the
                                                    documentation
        :param feature_binning:                     The strategy that should be used to assign examples to bins based on
                                                    their feature values. Must be 'auto', 'equal-width',
                                                    'equal-frequency' or 'none', if no feature binning should be used.
                                                    If set to 'auto', the most suitable strategy is chosen
                                                    automatically, depending on the characteristics of the feature
                                                    matrix. For additional options refer to the documentation
        :param label_binning:                       The strategy that should be used to assign labels to bins. Must be
                                                    'auto', 'equal-width' or 'none', if no label binning should be used.
                                                    If set to 'auto', the most suitable strategy is chosen
                                                    automatically, depending on the loss function and the type of rule
                                                    heads. For additional options refer to the documentation
        :param rule_pruning:                        The strategy that should be used to prune individual rules. Must be
                                                    'irep' or 'none', if no pruning should be used
        :param shrinkage:                           The shrinkage parameter, a.k.a. the "learning rate", that should be
                                                    used to shrink the weight of individual rules. Must be in (0, 1]
        :param l1_regularization_weight:            The weight of the L1 regularization. Must be at least 0
        :param l2_regularization_weight:            The weight of the L2 regularization. Must be at least 0
        :param parallel_rule_refinement:            Whether potential refinements of rules should be searched for in
                                                    parallel or not. Must be 'true', 'false' or 'auto', if the most
                                                    suitable strategy should be chosen automatically depending on the
                                                    loss function. For additional options refer to the documentation
        :param parallel_statistic_update:           Whether the gradients and Hessians for different examples should be
                                                    updated in parallel or not. Must be 'true', 'false' or 'auto', if
                                                    the most suitable strategy should be chosen automatically, depending
                                                    on the loss function. For additional options refer to the
                                                    documentation
        :param parallel_prediction:                 Whether predictions for different examples should be obtained in
                                                    parallel or not. Must be 'true' or 'false'. For additional options
                                                    refer to the documentation
        """
        super().__init__(feature_format, output_format, prediction_format)
        self.random_state = random_state
        self.statistic_format = statistic_format
        self.statistic_type = statistic_type
        self.default_rule = default_rule
        self.rule_induction = rule_induction
        self.max_rules = max_rules
        self.time_limit = time_limit
        self.global_pruning = global_pruning
        self.post_optimization = post_optimization
        self.head_type = head_type
        self.loss = loss
        self.marginal_probability_calibration = marginal_probability_calibration
        self.joint_probability_calibration = joint_probability_calibration
        self.binary_predictor = binary_predictor
        self.probability_predictor = probability_predictor
        self.output_sampling = output_sampling
        self.instance_sampling = instance_sampling
        self.feature_sampling = feature_sampling
        self.holdout = holdout
        self.feature_binning = feature_binning
        self.label_binning = label_binning
        self.rule_pruning = rule_pruning
        self.shrinkage = shrinkage
        self.l1_regularization_weight = l1_regularization_weight
        self.l2_regularization_weight = l2_regularization_weight
        self.parallel_rule_refinement = parallel_rule_refinement
        self.parallel_statistic_update = parallel_statistic_update
        self.parallel_prediction = parallel_prediction

    def _create_learner(self) -> Any:
        config = BoomerClassifierConfig()
        configure_rule_learner(self, config, BOOMER_CLASSIFIER_PARAMETERS)
        return BoomerClassifierWrapper(config)


class BoomerRegressor(RegressionRuleLearner, RegressorMixin):
    """
    A scikit-learn implementation of "BOOMER", an algorithm for learning gradient boosted multi-output rules, that can
    be used in regression problems.
    """

    def __init__(self,
                 random_state: Optional[int] = None,
                 feature_format: Optional[str] = None,
                 output_format: Optional[str] = None,
                 prediction_format: Optional[str] = None,
                 statistic_format: Optional[str] = None,
                 statistic_type: Optional[str] = None,
                 default_rule: Optional[str] = None,
                 rule_induction: Optional[str] = None,
                 max_rules: Optional[int] = None,
                 time_limit: Optional[int] = None,
                 global_pruning: Optional[str] = None,
                 post_optimization: Optional[str] = None,
                 head_type: Optional[str] = None,
                 loss: Optional[str] = None,
                 output_sampling: Optional[str] = None,
                 instance_sampling: Optional[str] = None,
                 feature_sampling: Optional[str] = None,
                 holdout: Optional[str] = None,
                 feature_binning: Optional[str] = None,
                 rule_pruning: Optional[str] = None,
                 shrinkage: Optional[float] = 0.3,
                 l1_regularization_weight: Optional[float] = None,
                 l2_regularization_weight: Optional[float] = None,
                 parallel_rule_refinement: Optional[str] = None,
                 parallel_statistic_update: Optional[str] = None,
                 parallel_prediction: Optional[str] = None):
        """
        :param random_state:                The seed to be used by RNGs. Must be at least 1
        :param statistic_format:            The format to be used for representing gradients and Hessians. Must be
                                            'dense', 'sparse' or 'auto', if the most suitable format should be chosen
                                            automatically
        :param statistic_type:              The data type to be used for representing gradients and Hessians. Must be
                                            '32-bit' or '64-bit'
        :param default_rule:                Whether a default rule should be induced or not. Must be 'true', 'false' or
                                            'auto', if it should be decided automatically whether a default rule should
                                            be induced or not
        :param rule_induction:              The algorithm that should be used for the induction of individual rules.
                                            Must be 'top-down-greedy' or 'top-down-beam-search'. For additional options
                                            refer to the documentation
        :param max_rules:                   The maximum number of rules to be learned (including the default rule). Must
                                            be at least 1 or 0, if the number of rules should not be restricted
        :param time_limit:                  The duration in seconds after which the induction of rules should be
                                            canceled. Must be at least 1 or 0, if no time limit should be set
        :param global_pruning:              The strategy that should be used for pruning entire rules. Must be
                                            'pre-pruning', 'post-pruning' or 'none', if no pruning should be used. For
                                            additional options refer to the documentation
        :param post_optimization:           Whether each rule in a previously learned model should be optimized by being
                                            relearned in the context of the other rules or not. Must be 'true' or
                                            'false'. For additional options refer to the documentation
        :param head_type:                   The type of the rule heads that should be used. Must be 'single',
                                            'complete', 'partial-fixed', 'partial-dynamic' or 'auto', if the type of the
                                            heads should be chosen automatically. For additional options refer to the
                                            documentation
        :param loss:                        The loss function to be minimized. Must be 'squared-error-decomposable' or
                                            'squared-error-non-decomposable'
        :param output_sampling:             The strategy that should be used to sample from the available outputs
                                            whenever a new rule is learned. Must be 'round-robin', 'without-replacement'
                                            or 'none', if no sampling should be used. For additional options refer to
                                            the documentation
        :param instance_sampling:           The strategy that should be used to sample from the available the training
                                            examples whenever a new rule is learned. Must be 'with-replacement',
                                            'without-replacement' or 'none', if no sampling should be used. For
                                            additional options refer to the documentation
        :param feature_sampling:            The strategy that is used to sample from the available features whenever a
                                            rule is refined. Must be 'without-replacement' or 'none', if no sampling
                                            should be used. For additional options refer to the documentation
        :param holdout:                     The name of the strategy that should be used to create a holdout set. Must
                                            be 'random' or 'none', if no holdout set should be used. If set to 'auto',
                                            the most suitable strategy is chosen automatically depending on whether a
                                            holdout set is needed and depending on the loss function. For additional
                                            options refer to the documentation
        :param feature_binning:             The strategy that should be used to assign examples to bins based on their
                                            feature values. Must be 'auto', 'equal-width', 'equal-frequency' or 'none',
                                            if no feature binning should be used. If set to 'auto', the most suitable
                                            strategy is chosen automatically, depending on the characteristics of the
                                            feature matrix. For additional options refer to the documentation
        :param rule_pruning:                The strategy that should be used to prune individual rules. Must be 'irep'
                                            or 'none', if no pruning should be used
        :param shrinkage:                   The shrinkage parameter, a.k.a. the "learning rate", that should be used to
                                            shrink the weight of individual rules. Must be in (0, 1]
        :param l1_regularization_weight:    The weight of the L1 regularization. Must be at least 0
        :param l2_regularization_weight:    The weight of the L2 regularization. Must be at least 0
        :param parallel_rule_refinement:    Whether potential refinements of rules should be searched for in parallel or
                                            not. Must be 'true', 'false' or 'auto', if the most suitable strategy should
                                            be chosen automatically depending on the loss function. For additional
                                            options refer to the documentation
        :param parallel_statistic_update:   Whether the gradients and Hessians for different examples should be updated
                                            in parallel or not. Must be 'true', 'false' or 'auto', if the most suitable
                                            strategy should be chosen automatically, depending on the loss function. For
                                            additional options refer to the documentation
        :param parallel_prediction:         Whether predictions for different examples should be obtained in parallel or
                                            not. Must be 'true' or 'false'. For additional options refer to the
                                            documentation
        """
        super().__init__(feature_format, output_format, prediction_format)
        self.random_state = random_state
        self.statistic_format = statistic_format
        self.statistic_type = statistic_type
        self.default_rule = default_rule
        self.rule_induction = rule_induction
        self.max_rules = max_rules
        self.time_limit = time_limit
        self.global_pruning = global_pruning
        self.post_optimization = post_optimization
        self.head_type = head_type
        self.loss = loss
        self.output_sampling = output_sampling
        self.instance_sampling = instance_sampling
        self.feature_sampling = feature_sampling
        self.holdout = holdout
        self.feature_binning = feature_binning
        self.rule_pruning = rule_pruning
        self.shrinkage = shrinkage
        self.l1_regularization_weight = l1_regularization_weight
        self.l2_regularization_weight = l2_regularization_weight
        self.parallel_rule_refinement = parallel_rule_refinement
        self.parallel_statistic_update = parallel_statistic_update
        self.parallel_prediction = parallel_prediction

    def _create_learner(self) -> Any:
        config = BoomerRegressorConfig()
        configure_rule_learner(self, config, BOOMER_REGRESSOR_PARAMETERS)
        return BoomerRegressorWrapper(config)
