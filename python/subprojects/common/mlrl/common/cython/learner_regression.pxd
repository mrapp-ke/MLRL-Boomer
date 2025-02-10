from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from mlrl.common.cython._types cimport uint32
from mlrl.common.cython.example_weights cimport IExampleWeights
from mlrl.common.cython.feature_info cimport IFeatureInfo
from mlrl.common.cython.feature_matrix cimport IColumnWiseFeatureMatrix, IRowWiseFeatureMatrix
from mlrl.common.cython.learner cimport ITrainingResult
from mlrl.common.cython.output_space_info cimport IOutputSpaceInfo
from mlrl.common.cython.prediction cimport IScorePredictor
from mlrl.common.cython.regression_matrix cimport IRowWiseRegressionMatrix
from mlrl.common.cython.rule_model cimport IRuleModel


cdef extern from "mlrl/common/learner_regression.hpp" nogil:

    cdef cppclass IRegressionRuleLearner:

        # Functions:

        unique_ptr[ITrainingResult] fit(const IExampleWeights& exampleWeights, const IFeatureInfo& featureInfo,
                                        const IColumnWiseFeatureMatrix& featureMatrix,
                                        const IRowWiseRegressionMatrix& regressionMatrix) const

        bool canPredictScores(const IRowWiseFeatureMatrix&  featureMatrix, uint32 numLabels) const

        unique_ptr[IScorePredictor] createScorePredictor(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const IOutputSpaceInfo& outputSpaceInfo, uint32 numLabels) except +


cdef class RegressionRuleLearner:

    # Functions:

    cdef IRegressionRuleLearner* get_regression_rule_learner_ptr(self)
