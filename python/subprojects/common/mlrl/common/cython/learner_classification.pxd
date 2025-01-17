from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from mlrl.common.cython._types cimport uint32
from mlrl.common.cython.example_weights cimport IExampleWeights
from mlrl.common.cython.feature_info cimport IFeatureInfo
from mlrl.common.cython.feature_matrix cimport IColumnWiseFeatureMatrix, IRowWiseFeatureMatrix
from mlrl.common.cython.instance_sampling cimport IExampleWiseStratifiedInstanceSamplingConfig, \
    IOutputWiseStratifiedInstanceSamplingConfig
from mlrl.common.cython.label_matrix cimport IRowWiseLabelMatrix
from mlrl.common.cython.learner cimport ITrainingResult
from mlrl.common.cython.output_space_info cimport IOutputSpaceInfo
from mlrl.common.cython.partition_sampling cimport IExampleWiseStratifiedBiPartitionSamplingConfig, \
    IOutputWiseStratifiedBiPartitionSamplingConfig
from mlrl.common.cython.prediction cimport IBinaryPredictor, IProbabilityPredictor, IScorePredictor, \
    ISparseBinaryPredictor
from mlrl.common.cython.probability_calibration cimport IJointProbabilityCalibrationModel, \
    IMarginalProbabilityCalibrationModel
from mlrl.common.cython.rule_model cimport IRuleModel


cdef extern from "mlrl/common/learner_classification.hpp" nogil:

    cdef cppclass IClassificationRuleLearner:

        # Functions:

        unique_ptr[ITrainingResult] fit(const IExampleWeights& exampleWeights, const IFeatureInfo& featureInfo,
                                        const IColumnWiseFeatureMatrix& featureMatrix,
                                        const IRowWiseLabelMatrix& labelMatrix) const

        bool canPredictScores(const IRowWiseFeatureMatrix&  featureMatrix, uint32 numLabels) const

        unique_ptr[IScorePredictor] createScorePredictor(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const IOutputSpaceInfo& outputSpaceInfo, uint32 numLabels) except +

        bool canPredictProbabilities(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const

        unique_ptr[IProbabilityPredictor] createProbabilityPredictor(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const IOutputSpaceInfo& outputSpaceInfo,
            const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
            const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) except +

        bool canPredictBinary(const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const

        unique_ptr[IBinaryPredictor] createBinaryPredictor(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const IOutputSpaceInfo& outputSpaceInfo,
            const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
            const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) except +

        unique_ptr[ISparseBinaryPredictor] createSparseBinaryPredictor(
            const IRowWiseFeatureMatrix& featureMatrix, const IRuleModel& ruleModel,
            const IOutputSpaceInfo& outputSpaceInfo,
            const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
            const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) except +


    cdef cppclass IOutputWiseStratifiedInstanceSamplingMixin:

        # Functions:

        IOutputWiseStratifiedInstanceSamplingConfig& useOutputWiseStratifiedInstanceSampling()


    cdef cppclass IExampleWiseStratifiedInstanceSamplingMixin:

        # Functions:

        IExampleWiseStratifiedInstanceSamplingConfig& useExampleWiseStratifiedInstanceSampling()


    cdef cppclass IOutputWiseStratifiedBiPartitionSamplingMixin:

        # Functions:

        IOutputWiseStratifiedBiPartitionSamplingConfig& useOutputWiseStratifiedBiPartitionSampling()


    cdef cppclass IExampleWiseStratifiedBiPartitionSamplingMixin:

        # Functions:

        IExampleWiseStratifiedBiPartitionSamplingConfig& useExampleWiseStratifiedBiPartitionSampling()


cdef class ClassificationRuleLearner:

    # Functions:

    cdef IClassificationRuleLearner* get_classification_rule_learner_ptr(self)
