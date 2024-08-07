"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from abc import ABC, abstractmethod

from mlrl.seco.cython.heuristic import FMeasureConfig, MEstimateConfig
from mlrl.seco.cython.lift_function import KlnLiftFunctionConfig, PeakLiftFunctionConfig
from mlrl.seco.cython.stopping_criterion import CoverageStoppingCriterionConfig


class NoCoverageStoppingCriterionMixin(ABC):
    """
    Allows to configure a rule learner to not use any stopping criterion that stops the induction of rules as soon as
    the sum of the weights of the uncovered labels is smaller or equal to a certain threshold.
    """

    @abstractmethod
    def use_no_coverage_stopping_criterion(self):
        """
        Configures the rule learner to not use any stopping criterion that stops the induction of rules as soon as the
        sum of the weights of the uncovered labels is smaller or equal to a certain threshold.
        """
        pass


class CoverageStoppingCriterionMixin(ABC):
    """
    Allows to configure a rule learner to use a stopping criterion that stops the induction of rules as soon as the sum
    of the weights of the uncovered labels is smaller or equal to a certain threshold.
    """

    @abstractmethod
    def use_coverage_stopping_criterion(self) -> CoverageStoppingCriterionConfig:
        """
        Configures the rule learner to use a stopping criterion that stops the induction of rules as soon as the sum of
        the weights of the uncovered labels is smaller or equal to a certain threshold.

        :return: A `CoverageStoppingCriterionConfig` that allows further configuration of the stopping criterion
        """
        pass
    

class SingleOutputHeadMixin(ABC):
    """
    Allows to configure a rule learner to induce rules with single-output heads that predict for a single output.
    """

    @abstractmethod
    def use_single_output_heads(self):
        """
        Configures the rule learner to induce rules with single-output heads that predict for a single output.
        """
        pass


class PartialHeadMixin(ABC):
    """
    Allows to configure a rule learner to induce rules with partial heads.
    """

    @abstractmethod
    def use_partial_heads(self):
        """
        Configures the rule learner to induce rules with partial heads that predict for a subset of the available
        labels.
        """
        pass


class NoLiftFunctionMixin(ABC):
    """
    Allows to configure a rule learner to not use a lift function.
    """

    @abstractmethod
    def use_no_lift_function(self):
        """
        Configures the rule learner to not use a lift function.
        """
        pass


class PeakLiftFunctionMixin(ABC):
    """
    Allows to configure a rule learner to use a lift function that monotonously increases until a certain number of
    labels, where the maximum lift is reached, and monotonously decreases afterwards.
    """

    @abstractmethod
    def use_peak_lift_function(self) -> PeakLiftFunctionConfig:
        """
        Configures the rule learner to use a lift function that monotonously increases until a certain number of labels,
        where the maximum lift is reached, and monotonously decreases afterwards.

        :return: A `PeakLiftFunctionConfig` that allows further configuration of the lift function
        """
        pass


class KlnLiftFunctionMixin(ABC):
    """
    Allows to configure a rule learner to use a lift function that monotonously increases according to the natural
    logarithm of the number of labels for which a rule predicts.
    """

    @abstractmethod
    def use_kln_lift_function(self) -> KlnLiftFunctionConfig:
        """
        Configures the rule learner to use a lift function that monotonously increases according to the natural
        logarithm of the number of labels for which a rule predicts.

        :return: A `KlnLiftFunctionConfig` that allows further configuration of the lift function
        """
        pass


class AccuracyHeuristicMixin(ABC):
    """
    Allows to configure a rule learner to use the "Accuracy" heuristic for learning rules.
    """

    @abstractmethod
    def use_accuracy_heuristic(self):
        """
        Configures the rule learner to use the "Accuracy" heuristic for learning rules.
        """
        pass


class AccuracyPruningHeuristicMixin(ABC):
    """
    Allows to configure a rule learner to use the "Accuracy" heuristic for pruning rules.
    """
    
    @abstractmethod
    def use_accuracy_pruning_heuristic(self):
        """
        Configures the rule learner to use the "Accuracy" heuristic for pruning rules.
        """
        pass


class FMeasureHeuristicMixin(ABC):
    """
    Allows to configure a rule learner to use the "F-Measure" heuristic for learning rules.
    """

    @abstractmethod
    def use_f_measure_heuristic(self) -> FMeasureConfig:
        """
        Configures the rule learner to use the "F-Measure" heuristic for learning rules.

        :return: A `FMeasureConfig` that allows further configuration of the heuristic
        """
        pass


class FMeasurePruningHeuristicMixin(ABC):
    """
    Allows to configure a rule learner to use the "F-Measure" heuristic for pruning rules.
    """

    @abstractmethod
    def use_f_measure_pruning_heuristic(self) -> FMeasureConfig:
        """
        Configures the rule learner to use the "F-Measure" heuristic for pruning rules.

        :return: A `FMeasureConfig` that allows further configuration of the heuristic
        """
        pass


class MEstimateHeuristicMixin(ABC):
    """
    Allows to configure a rule learner to use the "M-Estimate" heuristic for learning rules.
    """

    @abstractmethod
    def use_m_estimate_heuristic(self) -> MEstimateConfig:
        """
        Configures the rule learner to use the "M-Estimate" heuristic for learning rules.

        :return: A `MEstimateConfig` that allows further configuration of the heuristic
        """
        pass


class MEstimatePruningHeuristicMixin(ABC):
    """
    Allows to configure a rule learner to use the "M-Estimate" heuristic for pruning rules.
    """

    @abstractmethod
    def use_m_estimate_pruning_heuristic(self) -> MEstimateConfig:
        """
        Configures the rule learner to use the "M-Estimate" heuristic for pruning rules.

        :return: A `MEstimateConfig` that allows further configuration of the heuristic
        """
        pass


class LaplaceHeuristicMixin(ABC):
    """
    Allows to configure a rule learner to use the "Laplace" heuristic for learning rules.
    """

    @abstractmethod
    def use_laplace_heuristic(self):
        """
        Configures the rule learner to use the "Laplace" heuristic for learning rules.
        """
        pass


class LaplacePruningHeuristicMixin(ABC):
    """
    Allows to configure a rule learner to use the "Laplace" heuristic for pruning rules.
    """

    @abstractmethod
    def use_laplace_pruning_heuristic(self):
        """
        Configures the rule learner to use the "Laplace" heuristic for pruning rules.
        """
        pass


class PrecisionHeuristicMixin(ABC):
    """
    Allows to configure a rule learner to use the "Precision" heuristic for learning rules.
    """

    @abstractmethod
    def use_precision_heuristic(self):
        """
        Configures the rule learner to use the "Precision" heuristic for learning rules.
        """
        pass


class PrecisionPruningHeuristicMixin(ABC):
    """
    Allows to configure a rule learner to use the "Precision" heuristic for pruning rules.
    """

    @abstractmethod
    def use_precision_pruning_heuristic(self):
        """
        Configures the rule learner to use the "Precision" heuristic for pruning rules.
        """
        pass


class RecallHeuristicMixin(ABC):
    """
    Allows to configure a rule learner to use the "Recall" heuristic for pruning rules.
    """

    @abstractmethod
    def use_recall_heuristic(self):
        """
        Configures the rule learner to use the "Recall" heuristic for learning rules.
        """
        pass


class RecallPruningHeuristicMixin(ABC):
    """
    Allows to configure a rule learner to use the "Recall" heuristic for pruning rules.
    """

    @abstractmethod
    def use_recall_pruning_heuristic(self):
        """
        Configures the rule learner to use the "Recall" heuristic for pruning rules.
        """
        pass


class WraHeuristicMixin(ABC):
    """
    Allows to configure a rule learner to use the "Weighted Relative Accuracy" (WRA) heuristic for learning rules.
    """

    @abstractmethod
    def use_wra_heuristic(self):
        """
        Configures the rule learner to use the "Weighted Relative Accuracy" heuristic for learning rules.
        """
        pass


class WraPruningHeuristicMixin(ABC):
    """
    Allows to configure a rule learner to use the "Weighted Relative Accuracy" (WRA) heuristic for pruning rules.
    """

    @abstractmethod
    def use_wra_pruning_heuristic(self):
        """
        Configures the rule learner to use the "Weighted Relative Accuracy" heuristic for pruning rules.
        """
        pass


class OutputWiseBinaryPredictionMixin(ABC):
    """
    Allows to configure a rule learner to use a predictor for predicting whether individual labels of given query
    examples are relevant or irrelevant by processing rules of an existing rule-based model in the order they have been
    learned. If a rule covers an example, its prediction is applied to each label individually.
    """

    @abstractmethod
    def use_output_wise_binary_predictor(self):
        """
        Configures the rule learner to use predictor for predicting whether individual labels of given query examples
        are relevant or irrelevant by processing rules of an existing rule-based model in the order they have been
        learned. If a rule covers an example, its prediction is applied to each label individually.
        """
        pass
