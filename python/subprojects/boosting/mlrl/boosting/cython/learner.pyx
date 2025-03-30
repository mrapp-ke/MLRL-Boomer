"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from abc import ABC, abstractmethod

from mlrl.boosting.cython.head_type import DynamicPartialHeadConfig, FixedPartialHeadConfig
from mlrl.boosting.cython.post_processor import ConstantShrinkageConfig
from mlrl.boosting.cython.regularization import ManualRegularizationConfig


class AutomaticFeatureBinningMixin(ABC):
    """
    Allows to configure a rule learner to automatically decide whether a method for the assignment of numerical feature
    values to bins should be used or not.
    """

    @abstractmethod
    def use_automatic_feature_binning(self):
        """
        Configures the rule learning to automatically decide whether a method for the assignment of numerical feature
        values to bins should be used or not.
        """
             
             
class AutomaticParallelRuleRefinementMixin(ABC):
    """
    Allows to configure a rule learner to automatically decide whether multi-threading should be used for the parallel
    refinement of rules or not.
    """

    @abstractmethod
    def use_automatic_parallel_rule_refinement(self):
        """
        Configures the rule learner to automatically decide whether multi-threading should be used for the parallel
        refinement of rules or not.
        """
             
             
class AutomaticParallelStatisticUpdateMixin(ABC):
    """
    Allows to configure a rule learner to automatically decide whether multi-threading should be used for the parallel
    update of statistics or not.
    """

    @abstractmethod
    def use_automatic_parallel_statistic_update(self):
        """
        Configures the rule learner to automatically decide whether multi-threading should be used for the parallel
        update of statistics or not.
        """
             
             
class ConstantShrinkageMixin(ABC):
    """
    Allows to configure a rule learner to use a post processor that shrinks the weights fo rules by a constant
    "shrinkage" parameter.
    """

    @abstractmethod
    def use_constant_shrinkage_post_processor(self) -> ConstantShrinkageConfig:
        """
        Configures the rule learner to use a post-processor that shrinks the weights of rules by a constant "shrinkage"
        parameter.

        :return: A `ConstantShrinkageConfig` that allows further configuration of the post-processor
        """
             
             
class NoL1RegularizationMixin(ABC):
    """
    Allows to configure a rule learner to not use L1 regularization.
    """

    @abstractmethod
    def use_no_l1_regularization(self):
        """
        Configures the rule learner to not use L1 regularization.
        """


class L1RegularizationMixin(ABC):
    """
    Allows to configure a rule learner to use L1 regularization.
    """

    @abstractmethod
    def use_l1_regularization(self) -> ManualRegularizationConfig:
        """
        Configures the rule learner to use L1 regularization.

        :return: A `ManualRegularizationConfig` that allows further configuration of the regularization term
        """
            
             
class NoL2RegularizationMixin(ABC):
    """
    Allows to configure a rule learner to not use L2 regularization.
    """

    @abstractmethod
    def use_no_l2_regularization(self):
        """
        Configures the rule learner to not use L2 regularization.
        """
            
             
class L2RegularizationMixin(ABC):
    """
    Allows to configure a rule learner to use L2 regularization.
    """

    @abstractmethod
    def use_l2_regularization(self) -> ManualRegularizationConfig:
        """
        Configures the rule learner to use L2 regularization.

        :return: A `ManualRegularizationConfig` that allows further configuration of the regularization term
        """
             
             
class CompleteHeadMixin(ABC):
    """
    Allows to configure a rule learner to induce rules with complete heads that predict for all available outputs.
    """

    @abstractmethod
    def use_complete_heads(self):
        """
        Configures the rule learner to induce rules with complete heads that predict for all available outputs.
        """
            
             
class FixedPartialHeadMixin(ABC):
    """
    Allows to configure a rule learner to induce rules with partial heads that predict for a predefined number of
    outputs.
    """

    @abstractmethod
    def use_fixed_partial_heads(self) -> FixedPartialHeadConfig:
        """
        Configures the rule learner to induce rules with partial heads that predict for a predefined number of outputs.

        :return: A `FixedPartialHeadConfig` that allows further configuration of the rule heads
        """
            
             
class DynamicPartialHeadMixin(ABC):
    """
    Allows to configure a rule learner to induce rules with partial heads that predict for a subset of the available
    outputs that is determined dynamically.
    """

    @abstractmethod
    def use_dynamic_partial_heads(self) -> DynamicPartialHeadConfig:
        """
        Configures the rule learner to induce rules with partial heads that predict for a subset of the available
        outputs that is determined dynamically. Only those outputs for which the square of the predictive quality
        exceeds a certain threshold are included in a rule head.

        :return: A `DynamicPartialHeadConfig` that allows further configuration of the rule heads
        """
            
             
class SingleOutputHeadMixin(ABC):
    """
    Allows to configure a rule learner to induce rules with single-output heads that predict for a single output.
    """

    @abstractmethod
    def use_single_output_heads(self):
        """
        Configures the rule learner to induce rules with single-output heads that predict for a single output.
        """
            
             
class AutomaticHeadMixin(ABC):
    """
    Allows to configure a rule learner to automatically decide for the type of rule heads that should be used.
    """

    @abstractmethod
    def use_automatic_heads(self):
        """
        Configures the rule learner to automatically decide for the type of rule heads to be used.
        """
            
             
class NonDecomposableSquaredErrorLossMixin(ABC):
    """
    Allows to configure a rule learner to use a loss function that implements a multivariate variant of the squared
    error loss that is non-decomposable.
    """

    @abstractmethod
    def use_non_decomposable_squared_error_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multivariant variant of the squared error
        loss that is non-decomposable.
        """
            
             
class DecomposableSquaredErrorLossMixin(ABC):
    """
    Allows to configure a rule learner to use a loss function that implements a multivariate variant of the squared
    error loss that is decomposable.
    """

    @abstractmethod
    def use_decomposable_squared_error_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multivariate variant of the squared error
        loss that is decomposable.
        """
            
             
class OutputWiseScorePredictorMixin(ABC):
    """
    Allows to configure a rule learner to use a predictor that predicts output-wise scores for given query examples by
    summing up the scores that are provided by individual rules for each output individually.
    """

    @abstractmethod
    def use_output_wise_score_predictor(self):
        """
        Configures the rule learner to use a predictor that predict output-wise scores for given query examples by
        summing up the scores that are provided by individual rules for each output individually.
        """
