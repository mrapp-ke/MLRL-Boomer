"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from abc import ABC, abstractmethod

from mlrl.boosting.cython.label_binning import EqualWidthLabelBinningConfig
from mlrl.boosting.cython.prediction import ExampleWiseBinaryPredictorConfig, GfmBinaryPredictorConfig, \
    MarginalizedProbabilityPredictorConfig, OutputWiseBinaryPredictorConfig, OutputWiseProbabilityPredictorConfig
from mlrl.boosting.cython.probability_calibration import IsotonicJointProbabilityCalibratorConfig, \
    IsotonicMarginalProbabilityCalibratorConfig


class AutomaticPartitionSamplingMixin(ABC):
    """
    Allows to configure a rule learner to automatically decide whether a holdout set should be used or not.
    """

    @abstractmethod
    def use_automatic_partition_sampling(self):
        """
        Configures the rule learner to automatically decide whether a holdout set should be used or not.
        """

             
class NoDefaultRuleMixin(ABC):
    """
    Allows to configure a rule learner to not induce a default rule.
    """

    @abstractmethod
    def use_no_default_rule(self):
        """
        Configures the rule learner to not induce a default rule.
        """
             
             
class AutomaticDefaultRuleMixin(ABC):
    """
    Allows to configure a rule learner to automatically decide whether a default rule should be induced or not.
    """

    @abstractmethod
    def use_automatic_default_rule(self):
        """
        Configures the rule learner to automatically decide whether a default rule should be induced or not.
        """
            
             
class DenseStatisticsMixin(ABC):
    """
    Allows to configure a rule learner to use a dense representation of gradients and Hessians.
    """

    @abstractmethod
    def use_dense_statistics(self):
        """
        Configures the rule learner to use a dense representation of gradients and Hessians.
        """
            
             
class SparseStatisticsMixin(ABC):
    """
    Allows to configure a rule learner to use a sparse representation of gradients and Hessians, if possible.
    """

    @abstractmethod
    def use_sparse_statistics(self):
        """
        Configures the rule learner to use a sparse representation of gradients and Hessians, if possible.
        """
            
             
class AutomaticStatisticsMixin(ABC):
    """
    Allows to configure a rule learner to automatically decide whether a dense or sparse representation of gradients and
    Hessians should be used.
    """

    @abstractmethod
    def use_automatic_statistics(self):
        """
        Configures the rule learner to automatically decide whether a dense or sparse representation of gradients and
        Hessians should be used.
        """
            
             
class NonDecomposableLogisticLossMixin(ABC):
    """
    Allows to configure a rule learner to use a loss function that implements a multivariate variant of the logistic
    loss that is non-decomposable.
    """

    @abstractmethod
    def use_non_decomposable_logistic_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multivariate variant of the logistic loss
        that is non-decomposable.
        """
            
             
class NonDecomposableSquaredHingeLossMixin(ABC):
    """
    Allows to configure a rule learner to use a loss function that implements a multivariate variant of the squared
    hinge loss that is non-decomposable.
    """

    @abstractmethod
    def use_non_decomposable_squared_hinge_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multivariate variant of the squared hinge
        loss that is non-decomposable.
        """
            
             
class DecomposableLogisticLossMixin(ABC):
    """
    Allows to configure a rule learner to use a loss function that implements a multivariate variant of the logistic
    loss that is decomposable.
    """

    @abstractmethod
    def use_decomposable_logistic_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multivariate variant of the logistic loss
        that is decomposable.
        """
            
             
class DecomposableSquaredHingeLossMixin(ABC):
    """
    Allows to configure a rule learner to use a loss function that implements a multivariate variant of the squared
    hinge loss that is decomposable.
    """

    @abstractmethod
    def use_decomposable_squared_hinge_loss(self):
        """
        Configures the rule learner to use a loss function that implements a multivariate variant of the squared hinge
        loss that is decomposable.
        """
            
            
class NoLabelBinningMixin(ABC):
    """
    Allows to configure a rule learner to not use any method for the assignment of labels to bins.
    """

    @abstractmethod
    def use_no_label_binning(self):
        """
        Configures the rule learner to not use any method for the assignment of labels to bins.
        """
            
             
class EqualWidthLabelBinningMixin(ABC):
    """
    Allows to configure a rule learner to use a method for the assignment of labels to bins.
    """

    @abstractmethod
    def use_equal_width_label_binning(self) -> EqualWidthLabelBinningConfig:
        """
        Configures the rule learner to use a method for the assignment of labels to bins in a way such that each bin
        contains labels for which the predicted score is expected to belong to the same value range.

        :return: A `EqualWidthLabelBinningConfig` that allows further configuration of the method for the assignment of
                 labels to bins
        """
            
             
class AutomaticLabelBinningMixin(ABC):
    """
    Allows to configure a rule learner to automatically decide whether a method for the assignment of labels to bins
    should be used or not.
    """

    @abstractmethod
    def use_automatic_label_binning(self):
        """
        Configures the rule learner to automatically decide whether a method for the assignment of labels to bins should
        be used or not.
        """
            
             
class IsotonicMarginalProbabilityCalibrationMixin(ABC):
    """
    Allows to configure a rule learner to calibrate marginal probabilities via isotonic regression.
    """

    @abstractmethod
    def use_isotonic_marginal_probability_calibration(self) -> IsotonicMarginalProbabilityCalibratorConfig:
        """
        Configures the rule learner to calibrate marginal probabilities via isotonic regression.

        :return: An `IsotonicMarginalProbabilityCalibratorConfig` that allows further configuration of the calibrator
        """


class IsotonicJointProbabilityCalibrationMixin(ABC):
    """
    Allows to configure a rule learner to calibrate joint probabilities via isotonic regression.
    """

    @abstractmethod
    def use_isotonic_joint_probability_calibration(self) -> IsotonicJointProbabilityCalibratorConfig:
        """
        Configures the rule learner to calibrate joint probabilities via isotonic regression.

        :return: An `IsotonicJointProbabilityCalibratorConfig` that allows further configuration of the calibrator
        """


class OutputWiseProbabilityPredictorMixin(ABC):
    """
    Allows to configure a rule learner to use a predictor that predicts label-wise probabilities for given query
    examples by transforming the individual scores that are predicted for each label into probabilities.
    """

    @abstractmethod
    def use_output_wise_probability_predictor(self) -> OutputWiseProbabilityPredictorConfig:
        """
        Configures the rule learner to use a predictor that predicts label-wise probabilities for given query examples
        by transforming the individual scores that are predicted for each label into probabilities.

        :return: A `OutputWiseProbabilityPredictorConfig` that allows further configuration of the predictor
        """
            
             
class MarginalizedProbabilityPredictorMixin(ABC):
    """
    Allows to configure a rule learner to use predictor for predicting probability estimates by summing up the scores
    that are provided by individual rules of an existing rule-based model and comparing the aggregated score vector to
    the known label vectors according to a certain distance measure.
    """

    @abstractmethod
    def use_marginalized_probability_predictor(self) -> MarginalizedProbabilityPredictorConfig:
        """
        Configures the rule learner to use a predictor for predicting probability estimates by summing up the scores
        that are provided by individual rules of an existing rule-based model and comparing the aggregated score vector
        to the known label vectors according to a certain distance measure. The probability for an individual label
        calculates as the sum of the distances that have been obtained for all label vectors, where the respective label
        is specified to be relevant, divided by the total sum of all distances.

        :return: A `MarginalizedProbabilityPredictorConfig` that allows further configuration of the predictor
        """
            
             
class AutomaticProbabilityPredictorMixin(ABC):
    """
    Allows to configure a rule learner to automatically decide for a predictor for predicting probability estimates.
    """
    
    @abstractmethod
    def use_automatic_probability_predictor(self):
        """
        Configures the rule learner to automatically decide for a predictor for predicting probability estimates.
        """


class OutputWiseBinaryPredictorMixin(ABC):
    """
    Allows to configure a predictor that predicts whether individual labels of given query examples are relevant or
    irrelevant by discretizing the scores or probability estimates that are predicted for each label individually.
    """

    @abstractmethod
    def use_output_wise_binary_predictor(self) -> OutputWiseBinaryPredictorConfig:
        """
        Configures the rule learner to use a predictor that predicts whether individual labels of given query examples
        are relevant or irrelevant by discretizing the scores or probability estimates that are predicted for each label
        individually.

        :return: A `OutputWiseBinaryPredictorConfig` that allows further configuration of the predictor
        """
            
             
class ExampleWiseBinaryPredictorMixin(ABC):
    """
    Allows to configure a rule learner to use a predictor that predicts known label vectors for given query examples by
    comparing the predicted scores or probability estimates to the label vectors encountered in the training data.
    """
            
    @abstractmethod
    def use_example_wise_binary_predictor(self) -> ExampleWiseBinaryPredictorConfig:
        """
        Configures the rule learner to use a predictor that predicts known label vectors for given query examples by
        comparing the predicted scores or probability estimates to the label vectors encountered in the training data.

        :return: An `ExampleWiseBinaryPredictorConfig` that allows further configuration of the predictor
        """


class GfmBinaryPredictorMixin(ABC):
    """
    Allows to configure a rule learner to use a predictor that predicts whether individual labels of given query
    examples are relevant or irrelevant by discretizing the scores or probability estimates that are predicted for each
    label according to the general F-measure maximizer (GFM).
    """

    @abstractmethod
    def use_gfm_binary_predictor(self) -> GfmBinaryPredictorConfig:
        """
        Configures the rule learner to use a predictor that predicts whether individual labels of given query examples
        are relevant or irrelevant by discretizing the scores or probability estimates that are predicted for each label
        according to the general F-measure maximizer (GFM).

        :return: A `GfmBinaryPredictorConfig` that allows further configuration of the predictor
        """
            
             
class AutomaticBinaryPredictorMixin(ABC):
    """
    Allows to configure a rule learner to automatically decide for a predictor for predicting whether individual labels
    are relevant or irrelevant.
    """

    @abstractmethod
    def use_automatic_binary_predictor(self):
        """
        Configures the rule learner to automatically decide for a predictor for predicting whether individual labels are
        relevant or irrelevant.
        """
