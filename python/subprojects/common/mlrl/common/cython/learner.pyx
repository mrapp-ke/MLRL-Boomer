"""
@author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from mlrl.common.cython._arrays cimport array_uint32, c_matrix_uint8, c_matrix_float64
from mlrl.common.cython.feature_matrix cimport ColumnWiseFeatureMatrix, RowWiseFeatureMatrix
from mlrl.common.cython.label_matrix cimport RowWiseLabelMatrix
from mlrl.common.cython.label_space_info cimport create_label_space_info
from mlrl.common.cython.nominal_feature_mask cimport NominalFeatureMask
from mlrl.common.cython.rule_model cimport create_rule_model

from libcpp.utility cimport move

from cython.operator cimport dereference

from scipy.sparse import csr_matrix
import numpy as np


cdef class TrainingResult:
    """
    A wrapper for the pure virtual C++ class `ITrainingResult`.
    """

    def __cinit__(self, uint32 num_labels, RuleModel rule_model not None, LabelSpaceInfo label_space_info not None):
        """
        :param num_labels:          The number of labels for which a model has been trained
        :param rule_model:          The `RuleModel` that has been trained
        :param label_space_info:    The `LabelSpaceInfo` that may be used as a basis for making predictions
        """
        self.num_labels = num_labels
        self.rule_model = rule_model
        self.label_space_info = label_space_info


cdef class RuleLearner:
    """
    A wrapper for the pure virtual C++ class `IRuleLearner`.
    """

    cdef IRuleLearner* get_rule_learner_ptr(self):
        pass

    def fit(self, NominalFeatureMask nominal_feature_mask not None, ColumnWiseFeatureMatrix feature_matrix not None,
            RowWiseLabelMatrix label_matrix not None, uint32 random_state) -> TrainingResult:
        """
        Applies the rule learner to given training examples and corresponding ground truth labels.

        :param nominal_feature_mask:    A `NominalFeatureMask` that allows to check whether individual features are
                                        nominal or not
        :param feature_matrix:          A `ColumnWiseFeatureMatrix` that provides column-wise access to the feature
                                        values of the training examples
        :param label_matrix:            A `RowWiseLabelMatrix` that provides row-wise access to the ground truth labels
                                        of the training examples
        :param random_state:            The seed to be used by random number generators
        :return:                        The `TrainingResult` that provides access to the result of fitting the rule
                                        learner to the training data
        """
        cdef unique_ptr[ITrainingResult] training_result_ptr = self.get_rule_learner_ptr().fit(
            dereference(nominal_feature_mask.get_nominal_feature_mask_ptr()),
            dereference(feature_matrix.get_column_wise_feature_matrix_ptr()),
            dereference(label_matrix.get_row_wise_label_matrix_ptr()), random_state)
        cdef uint32 num_labels = training_result_ptr.get().getNumLabels()
        cdef unique_ptr[IRuleModel] rule_model_ptr = move(training_result_ptr.get().getRuleModel())
        cdef unique_ptr[ILabelSpaceInfo] label_space_info_ptr = move(training_result_ptr.get().getLabelSpaceInfo())
        cdef RuleModel rule_model = create_rule_model(move(rule_model_ptr))
        cdef LabelSpaceInfo label_space_info = create_label_space_info(move(label_space_info_ptr))
        return TrainingResult.__new__(TrainingResult, num_labels, rule_model, label_space_info)

    def predict_labels(self, RowWiseFeatureMatrix feature_matrix not None, RuleModel rule_model not None,
                       LabelSpaceInfo label_space_info not None, uint32 num_labels) -> np.ndarray:
        """
        Obtains and returns dense predictions for given query examples.

        :param feature_matrix:      A `RowWiseFeatureMatrix` that provides row-wise access to the feature values of the
                                    query examples
        :param rule_model:          The `RuleModel` that should be used to obtain predictions
        :param label_space_info:    The `LabelSpaceInfo` that may be used as a basis for obtaining predictions
        :param num_labels:          The number of labels to predict for
        :return:                    A `numpy.ndarray` of type `uint8`, shape `(num_examples, num_labels)`, that stores
                                    the predictions
        """
        cdef unique_ptr[DensePredictionMatrix[uint8]] prediction_matrix_ptr = self.get_rule_learner_ptr().predictLabels(
            dereference(feature_matrix.get_row_wise_feature_matrix_ptr()),
            dereference(rule_model.get_rule_model_ptr()),
            dereference(label_space_info.get_label_space_info_ptr()),
            num_labels)
        cdef uint8* array = prediction_matrix_ptr.get().release()
        cdef uint32 num_examples = feature_matrix.get_feature_matrix_ptr().getNumRows()
        cdef uint8[:, ::1] prediction_matrix = c_matrix_uint8(array, num_examples, num_labels)
        return np.asarray(prediction_matrix)

    def predict_sparse_labels(self, RowWiseFeatureMatrix feature_matrix not None, RuleModel rule_model not None,
                              LabelSpaceInfo label_space_info not None, uint32 num_labels) -> csr_matrix:
        """
        Obtains and returns sparse predictions for given query examples.

        :param feature_matrix:      A `RowWiseFeatureMatrix` that provides row-wise access to the feature values of the
                                    query examples
        :param rule_model:          The `RuleModel` that should be used to obtain predictions
        :param label_space_info:    The `LabelSpaceInfo` that may be used as a basis for obtaining predictions
        :param num_labels:          The number of labels to predict for
        :return:                    A `scipy.sparse.csr_matrix` of type `uint8`, shape `(num_examples, num_labels)` that
                                    stores the predictions
        """
        cdef unique_ptr[BinarySparsePredictionMatrix] prediction_matrix_ptr = self.get_rule_learner_ptr().predictSparseLabels(
            dereference(feature_matrix.get_row_wise_feature_matrix_ptr()),
            dereference(rule_model.get_rule_model_ptr()),
            dereference(label_space_info.get_label_space_info_ptr()),
            num_labels)
        cdef uint32* row_indices = prediction_matrix_ptr.get().releaseRowIndices()
        cdef uint32* col_indices = prediction_matrix_ptr.get().releaseColIndices()
        cdef uint32 num_non_zero_elements = prediction_matrix_ptr.get().getNumNonZeroElements()
        cdef uint32 num_examples = feature_matrix.get_feature_matrix_ptr().getNumRows()
        data = np.ones(shape=(num_non_zero_elements), dtype=np.uint8) if num_non_zero_elements > 0 else np.asarray([])
        indices = np.asarray(array_uint32(col_indices, num_non_zero_elements) if num_non_zero_elements > 0 else [])
        indptr = np.asarray(array_uint32(row_indices, num_examples + 1))
        return csr_matrix((data, indices, indptr), shape=(num_examples, num_labels))

    def can_predict_probabilities(self) -> bool:
        """
        Returns whether the rule learner is able to predict probability estimates or not.

        :return: True, if the rule learner is able to predict probability estimates, False otherwise
        """
        return self.get_rule_learner_ptr().canPredictProbabilities()

    def predict_probabilities(self, RowWiseFeatureMatrix feature_matrix not None, RuleModel rule_model not None,
                              LabelSpaceInfo label_space_info not None, uint32 num_labels) -> np.ndarray:
        """
        Obtains and returns probability estimates for given query examples.

        :param feature_matrix:      A `RowWiseFeatureMatrix` that provides row-wise access to the feature values of the
                                    query examples
        :param rule_model:          The `RuleModel` that should be used to obtain predictions
        :param label_space_info:    The `LabelSpaceInfo` that may be used as a basis for obtaining predictions
        :param num_labels:          The number of labels to predict for
        :return:                    A `scipy.sparse.csr_matrix` of type `uint8`, shape `(num_examples, num_labels)` that
                                    stores the predictions
        """
        cdef unique_ptr[DensePredictionMatrix[float64]] prediction_matrix_ptr = self.get_rule_learner_ptr().predictProbabilities(
            dereference(feature_matrix.get_row_wise_feature_matrix_ptr()),
            dereference(rule_model.get_rule_model_ptr()),
            dereference(label_space_info.get_label_space_info_ptr()),
            num_labels)
        cdef float64* array = prediction_matrix_ptr.get().release()
        cdef uint32 num_examples = feature_matrix.get_feature_matrix_ptr().getNumRows()
        cdef float64[:, ::1] prediction_matrix = c_matrix_float64(array, num_examples, num_labels)
        return np.asarray(prediction_matrix)
