"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from cython.operator cimport dereference
from libcpp.utility cimport move

from mlrl.common.cython.example_weights cimport ExampleWeights
from mlrl.common.cython.feature_info cimport FeatureInfo
from mlrl.common.cython.feature_matrix cimport ColumnWiseFeatureMatrix, RowWiseFeatureMatrix
from mlrl.common.cython.label_matrix cimport RowWiseLabelMatrix
from mlrl.common.cython.learner cimport TrainingResult
from mlrl.common.cython.output_space_info cimport OutputSpaceInfo, create_output_space_info
from mlrl.common.cython.prediction cimport BinaryPredictor, ProbabilityPredictor, ScorePredictor, SparseBinaryPredictor
from mlrl.common.cython.probability_calibration cimport JointProbabilityCalibrationModel, \
    MarginalProbabilityCalibrationModel, create_joint_probability_calibration_model, \
    create_marginal_probability_calibration_model
from mlrl.common.cython.rule_model cimport RuleModel, create_rule_model

from abc import ABC, abstractmethod

from mlrl.common.cython.instance_sampling import ExampleWiseStratifiedInstanceSamplingConfig, \
    OutputWiseStratifiedInstanceSamplingConfig
from mlrl.common.cython.partition_sampling import ExampleWiseStratifiedBiPartitionSamplingConfig, \
    OutputWiseStratifiedBiPartitionSamplingConfig


cdef class ClassificationRuleLearner:
    """
    A rule learner that can be applied to classification problems.
    """

    cdef IClassificationRuleLearner* get_classification_rule_learner_ptr(self):
        pass

    def fit(self, ExampleWeights example_weights not None, FeatureInfo feature_info not None,
            ColumnWiseFeatureMatrix feature_matrix not None,
            RowWiseLabelMatrix label_matrix not None) -> TrainingResult:
        """
        Applies the rule learner to given training examples and corresponding ground truth labels.

        :param example_weights: `ExampleWeights` that provide access to the weights of individual training examples
        :param feature_info:    A `FeatureInfo` that provides information about the types of individual features
        :param feature_matrix:  A `ColumnWiseFeatureMatrix` that provides column-wise access to the feature values of
                                the training examples
        :param label_matrix:    A `RowWiseLabelMatrix` that provides row-wise access to the ground truth labels of the
                                training examples
        :return:                The `TrainingResult` that provides access to the result of fitting the rule learner to
                                the training data
        """
        cdef unique_ptr[ITrainingResult] training_result_ptr = self.get_classification_rule_learner_ptr().fit(
            dereference(example_weights.get_example_weights_ptr()),
            dereference(feature_info.get_feature_info_ptr()),
            dereference(feature_matrix.get_column_wise_feature_matrix_ptr()),
            dereference(label_matrix.get_row_wise_label_matrix_ptr()))
        cdef uint32 num_outputs = training_result_ptr.get().getNumOutputs()
        cdef unique_ptr[IRuleModel] rule_model_ptr = move(training_result_ptr.get().getRuleModel())
        cdef unique_ptr[IOutputSpaceInfo] output_space_info_ptr = move(training_result_ptr.get().getOutputSpaceInfo())
        cdef unique_ptr[IMarginalProbabilityCalibrationModel] marginal_probability_calibration_model_ptr = \
            move(training_result_ptr.get().getMarginalProbabilityCalibrationModel())
        cdef unique_ptr[IJointProbabilityCalibrationModel] joint_probability_calibration_model_ptr = \
            move(training_result_ptr.get().getJointProbabilityCalibrationModel())
        cdef RuleModel rule_model = create_rule_model(move(rule_model_ptr))
        cdef OutputSpaceInfo output_space_info = create_output_space_info(move(output_space_info_ptr))
        cdef MarginalProbabilityCalibrationModel marginal_probability_calibration_model = \
            create_marginal_probability_calibration_model(move(marginal_probability_calibration_model_ptr))
        cdef JointProbabilityCalibrationModel joint_probability_calibration_model = \
            create_joint_probability_calibration_model(move(joint_probability_calibration_model_ptr))
        return TrainingResult.__new__(TrainingResult, num_outputs, rule_model, output_space_info,
                                      marginal_probability_calibration_model, joint_probability_calibration_model)

    def can_predict_scores(self, RowWiseFeatureMatrix feature_matrix not None, uint32 num_labels) -> bool:
        """
        Returns whether the rule learner is able to predict scores or not.

        :param feature_matrix:  A `RowWiseFeatureMatrix` that provides row-wise access to the feature values of the
                                query examples
        :param num_labels:      The number of labels to predict for
        :return:                True, if the rule learner is able to predict scores, False otherwise
        """
        return self.get_classification_rule_learner_ptr().canPredictScores(
            dereference(feature_matrix.get_row_wise_feature_matrix_ptr()), num_labels)

    def create_score_predictor(self, RowWiseFeatureMatrix feature_matrix not None, RuleModel rule_model not None,
                               OutputSpaceInfo output_space_info not None, uint32 num_labels) -> ScorePredictor:
        """
        Creates and returns a predictor that may be used to predict scores for given query examples. If the prediction
        of scores is not supported by the rule learner, a `RuntimeError` is thrown.

        :param feature_matrix:      A `RowWiseFeatureMatrix` that provides row-wise access to the feature values of the
                                    query examples
        :param rule_model:          The `RuleModel` that should be used to obtain predictions
        :param output_space_info:   The `OutputSpaceInfo` that provides information about the output space that may be
                                    used as a basis for obtaining predictions
        :param num_labels:          The number of labels to predict for
        :return:                    A `ScorePredictor` that may be used to predict scores for the given query examples
        """
        cdef unique_ptr[IScorePredictor] predictor_ptr = \
            move(self.get_classification_rule_learner_ptr().createScorePredictor(
                dereference(feature_matrix.get_row_wise_feature_matrix_ptr()),
                dereference(rule_model.get_rule_model_ptr()),
                dereference(output_space_info.get_output_space_info_ptr()),
                num_labels))
        cdef ScorePredictor score_predictor = ScorePredictor.__new__(ScorePredictor)
        score_predictor.predictor_ptr = move(predictor_ptr)
        return score_predictor

    def can_predict_probabilities(self, RowWiseFeatureMatrix feature_matrix not None, uint32 num_labels) -> bool:
        """
        Returns whether the rule learner is able to predict probability estimates or not.

        :param feature_matrix:  A `RowWiseFeatureMatrix` that provides row-wise access to the feature values of the
                                query examples
        :param num_labels:      The number of labels to predict for
        :return:                True, if the rule learner is able to predict probability estimates, False otherwise
        """
        return self.get_classification_rule_learner_ptr().canPredictProbabilities(
            dereference(feature_matrix.get_row_wise_feature_matrix_ptr()), num_labels)

    def create_probability_predictor(self, RowWiseFeatureMatrix feature_matrix not None, RuleModel rule_model not None,
                                     OutputSpaceInfo output_space_info not None,
                                     MarginalProbabilityCalibrationModel marginal_probability_calibration_model not None,
                                     JointProbabilityCalibrationModel joint_probability_calibration_model not None,
                                     uint32 num_labels) -> ProbabilityPredictor:
        """
        Creates and returns a predictor that may be used to predict probability estimates for given query examples. If
        the prediction of probability estimates is not supported by the rule learner, a `RuntimeError` is thrown.

        :param feature_matrix:                          A `RowWiseFeatureMatrix` that provides row-wise access to the
                                                        feature values of the query examples
        :param rule_model:                              The `RuleModel` that should be used to obtain predictions
        :param output_space_info:                       The `OutputSpaceInfo` that provides information about the output
                                                        space that may be used as a basis for obtaining predictions
        :param marginal_probability_calibration_model:  The `MarginalProbabilityCalibrationModel` that may be used for
                                                        the calibration of marginal probabilities
        :param joint_probability_calibration_model:     The `JointProbabilityCalibrationModel` that may be used for the
                                                        calibration of joint probabilities    
        :param num_labels:                              The number of labels to predict for
        :return:                                        A `ProbabilityPredictor` that may be used to predict probability
                                                        estimates for the given query examples
        """
        cdef unique_ptr[IProbabilityPredictor] predictor_ptr = \
            move(self.get_classification_rule_learner_ptr().createProbabilityPredictor(
                dereference(feature_matrix.get_row_wise_feature_matrix_ptr()),
                dereference(rule_model.get_rule_model_ptr()),
                dereference(output_space_info.get_output_space_info_ptr()),
                dereference(marginal_probability_calibration_model.get_marginal_probability_calibration_model_ptr()),
                dereference(joint_probability_calibration_model.get_joint_probability_calibration_model_ptr()),
                num_labels))
        cdef ProbabilityPredictor probability_predictor = ProbabilityPredictor.__new__(ProbabilityPredictor)
        probability_predictor.predictor_ptr = move(predictor_ptr)
        return probability_predictor

    def can_predict_binary(self, RowWiseFeatureMatrix feature_matrix not None, uint32 num_labels) -> bool:
        """
        Returns whether the rule learner is able to predict binary labels or not.

        :param feature_matrix:  A `RowWiseFeatureMatrix` that provides row-wise access to the feature values of the
                                query examples
        :param num_labels:      The number of labels to predict for
        :return:                True, if the rule learner is able to predict binary labels, False otherwise
        """
        return self.get_classification_rule_learner_ptr().canPredictBinary(
            dereference(feature_matrix.get_row_wise_feature_matrix_ptr()), num_labels)

    def create_binary_predictor(self, RowWiseFeatureMatrix feature_matrix not None, RuleModel rule_model not None,
                                OutputSpaceInfo output_space_info not None,
                                MarginalProbabilityCalibrationModel marginal_probability_calibration_model not None,
                                JointProbabilityCalibrationModel joint_probability_calibration_model not None,
                                uint32 num_labels) -> BinaryPredictor:
        """
        Creates and returns a predictor that may be used to predict binary labels for given query examples. If the
        prediction of binary labels is not supported by the rule learner, a `RuntimeError` is thrown.

        :param feature_matrix:                          A `RowWiseFeatureMatrix` that provides row-wise access to the
                                                        feature values of the query examples
        :param rule_model:                              The `RuleModel` that should be used to obtain predictions
        :param output_space_info:                       The `OutputSpaceInfo` that provides information about the output
                                                        space that may be used as a basis for obtaining predictions
        :param marginal_probability_calibration_model:  The `MarginalProbabilityCalibrationModel` that may be used for
                                                        the calibration of marginal probabilities
        :param joint_probability_calibration_model:     The `JointProbabilityCalibrationModel` that may be used for the
                                                        calibration of joint probabilities    
        :param num_labels:                              The number of labels to predict for
        :return:                                        A `BinaryPredictor` that may be used to predict binary labels
                                                        for the given query examples
        """
        cdef unique_ptr[IBinaryPredictor] predictor_ptr = \
            move(self.get_classification_rule_learner_ptr().createBinaryPredictor(
                dereference(feature_matrix.get_row_wise_feature_matrix_ptr()),
                dereference(rule_model.get_rule_model_ptr()),
                dereference(output_space_info.get_output_space_info_ptr()),
                dereference(marginal_probability_calibration_model.get_marginal_probability_calibration_model_ptr()),
                dereference(joint_probability_calibration_model.get_joint_probability_calibration_model_ptr()),
                num_labels))
        cdef BinaryPredictor binary_predictor = BinaryPredictor.__new__(BinaryPredictor)
        binary_predictor.predictor_ptr = move(predictor_ptr)
        return binary_predictor

    def create_sparse_binary_predictor(self, RowWiseFeatureMatrix feature_matrix not None,
                                       RuleModel rule_model not None, OutputSpaceInfo output_space_info not None,
                                       MarginalProbabilityCalibrationModel marginal_probability_calibration_model not None,
                                       JointProbabilityCalibrationModel joint_probability_calibration_model not None,
                                       uint32 num_labels) -> SparseBinaryPredictor:
        """
        Creates and returns a predictor that may be used to predict sparse binary labels for given query examples. If
        the prediction of sparse binary labels is not supported by the rule learner, a `RuntimeError` is thrown.

        :param feature_matrix:                          A `RowWiseFeatureMatrix` that provides row-wise access to the
                                                        feature values of the query examples
        :param rule_model:                              The `RuleModel` that should be used to obtain predictions
        :param output_space_info:                       The `OutputSpaceInfo` that provides information about the output
                                                        space that may be used as a basis for obtaining predictions
        :param marginal_probability_calibration_model:  The `MarginalProbabilityCalibrationModel` that may be used for
                                                        the calibration of marginal probabilities
        :param joint_probability_calibration_model:     The `JointProbabilityCalibrationModel` that may be used for the
                                                        calibration of joint probabilities                                                            
        :param num_labels:                              The number of labels to predict for
        :return:                                        A `SparseBinaryPredictor` that may be used to predict sparse
                                                        binary labels for the given query examples
        """
        cdef unique_ptr[ISparseBinaryPredictor] predictor_ptr = \
            move(self.get_classification_rule_learner_ptr().createSparseBinaryPredictor(
                dereference(feature_matrix.get_row_wise_feature_matrix_ptr()),
                dereference(rule_model.get_rule_model_ptr()),
                dereference(output_space_info.get_output_space_info_ptr()),
                dereference(marginal_probability_calibration_model.get_marginal_probability_calibration_model_ptr()),
                dereference(joint_probability_calibration_model.get_joint_probability_calibration_model_ptr()),
                num_labels))
        cdef SparseBinaryPredictor sparse_binary_predictor = SparseBinaryPredictor.__new__(SparseBinaryPredictor)
        sparse_binary_predictor.predictor_ptr = move(predictor_ptr)
        return sparse_binary_predictor


class OutputWiseStratifiedInstanceSamplingMixin(ABC):
    """
    Allows to configure a rule learner to use label-wise stratified instance sampling.
    """

    @abstractmethod
    def use_output_wise_stratified_instance_sampling(self) -> OutputWiseStratifiedInstanceSamplingConfig:
        """
        Configures the rule learner to sample from the available training examples using stratification, such that for
        each label the proportion of relevant and irrelevant examples is maintained, whenever a new rule should be
        learned.

        :return: An `OutputWiseStratifiedInstanceSamplingConfig` that allows further configuration of the method for
                 sampling instances
        """


class ExampleWiseStratifiedInstanceSamplingMixin(ABC):
    """
    Allows to configure a rule learner to use example-wise stratified instance sampling.
    """

    @abstractmethod
    def use_example_wise_stratified_instance_sampling(self) -> ExampleWiseStratifiedInstanceSamplingConfig:
        """
        Configures the rule learner to sample from the available training examples using stratification, where distinct
        label vectors are treated as individual classes, whenever a new rule should be learned.

        :return: An `ExampleWiseStratifiedInstanceSamplingConfig` that allows further configuration of the method for
                 sampling instances
        """


class OutputWiseStratifiedBiPartitionSamplingMixin(ABC):
    """
    Allows to configure a rule learner to partition the available training examples into a training set and a holdout
    set using stratification, such that for each label the proportion of relevant and irrelevant examples is maintained.
    """

    @abstractmethod
    def use_output_wise_stratified_bi_partition_sampling(self) -> OutputWiseStratifiedBiPartitionSamplingConfig:
        """
        Configures the rule learner to partition the available training examples into a training set and a holdout set
        using stratification, such that for each label the proportion of relevant and irrelevant examples is maintained.

        :return: An `OutputWiseStratifiedBiPartitionSamplingConfig` that allows further configuration of the method for
                 partitioning the available training examples into a training and a holdout set
        """


class ExampleWiseStratifiedBiPartitionSamplingMixin(ABC):
    """
    Allows to configure a rule learner to partition the available training examples into a training set and a holdout
    set using stratification, where distinct label vectors are treated as individual classes.
    """

    @abstractmethod
    def use_example_wise_stratified_bi_partition_sampling(self) -> ExampleWiseStratifiedBiPartitionSamplingConfig:
        """
        Configures the rule learner to partition the available training examples into a training set and a holdout set
        using stratification, where distinct label vectors are treated as individual classes

        :return: An `ExampleWiseStratifiedBiPartitionSamplingConfig` that allows further configuration of the method for
                 partitioning the available training examples into a training and a holdout set
        """
