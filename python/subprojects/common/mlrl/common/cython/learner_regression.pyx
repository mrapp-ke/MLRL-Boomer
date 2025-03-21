"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from cython.operator cimport dereference
from libcpp.utility cimport move

from mlrl.common.cython.example_weights cimport ExampleWeights
from mlrl.common.cython.feature_info cimport FeatureInfo
from mlrl.common.cython.feature_matrix cimport ColumnWiseFeatureMatrix, RowWiseFeatureMatrix
from mlrl.common.cython.learner cimport TrainingResult
from mlrl.common.cython.output_space_info cimport OutputSpaceInfo, create_output_space_info
from mlrl.common.cython.prediction cimport ScorePredictor
from mlrl.common.cython.probability_calibration cimport IJointProbabilityCalibrationModel, \
    IMarginalProbabilityCalibrationModel, JointProbabilityCalibrationModel, MarginalProbabilityCalibrationModel, \
    create_joint_probability_calibration_model, create_marginal_probability_calibration_model
from mlrl.common.cython.regression_matrix cimport RowWiseRegressionMatrix
from mlrl.common.cython.rule_model cimport RuleModel, create_rule_model


cdef class RegressionRuleLearner:
    """
    A rule learner that can be applied to regression problems.
    """

    cdef IRegressionRuleLearner* get_regression_rule_learner_ptr(self):
        pass

    def fit(self, ExampleWeights example_weights not None, FeatureInfo feature_info not None,
            ColumnWiseFeatureMatrix feature_matrix not None,
            RowWiseRegressionMatrix regression_matrix not None) -> TrainingResult:
        """
        Applies the rule learner to given training examples and corresponding ground truth regression scores.
        
        :param example_weights:     `ExampleWeights` that provide access to the weights of individual training examples
        :param feature_info:        A reference to an object of type `IFeatureInfo` that provides information about the
                                    types of individual features
        :param feature_matrix:      A reference to an object of type `IColumnWiseFeatureMatrix` that provides
                                    column-wise access to the feature values of the training examples
        :param regression_matrix:   A reference to an object of type `IRowWiseRegressionMatrix` that provides row-wise
                                    access to the ground truth regression scores of the training examples
        :return:                    An unique pointer to an object of type `ITrainingResult` that provides access to the
                                    results of fitting the rule learner to the training data
        """
        cdef unique_ptr[ITrainingResult] training_result_ptr = self.get_regression_rule_learner_ptr().fit(
            dereference(example_weights.get_example_weights_ptr()),
            dereference(feature_info.get_feature_info_ptr()),
            dereference(feature_matrix.get_column_wise_feature_matrix_ptr()),
            dereference(regression_matrix.get_row_wise_regression_matrix_ptr()))
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
        return self.get_regression_rule_learner_ptr().canPredictScores(
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
            move(self.get_regression_rule_learner_ptr().createScorePredictor(
                dereference(feature_matrix.get_row_wise_feature_matrix_ptr()),
                dereference(rule_model.get_rule_model_ptr()),
                dereference(output_space_info.get_output_space_info_ptr()),
                num_labels))
        cdef ScorePredictor score_predictor = ScorePredictor.__new__(ScorePredictor)
        score_predictor.predictor_ptr = move(predictor_ptr)
        return score_predictor
