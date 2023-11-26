#include "mlrl/common/input/feature_matrix_c_contiguous.hpp"

#include "mlrl/common/prediction/predictor_binary.hpp"
#include "mlrl/common/prediction/predictor_probability.hpp"
#include "mlrl/common/prediction/predictor_score.hpp"

CContiguousFeatureMatrix::CContiguousFeatureMatrix(const float32* array, uint32 numRows, uint32 numCols)
    : CContiguousView<const float32>(array, numRows, numCols) {}

bool CContiguousFeatureMatrix::isSparse() const {
    return false;
}

uint32 CContiguousFeatureMatrix::getNumExamples() const {
    return Matrix::numRows;
}

uint32 CContiguousFeatureMatrix::getNumFeatures() const {
    return Matrix::numCols;
}

std::unique_ptr<IBinaryPredictor> CContiguousFeatureMatrix::createBinaryPredictor(
  const IBinaryPredictorFactory& factory, const IRuleModel& ruleModel, const ILabelSpaceInfo& labelSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return ruleModel.createBinaryPredictor(factory, *this, labelSpaceInfo, marginalProbabilityCalibrationModel,
                                           jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<ISparseBinaryPredictor> CContiguousFeatureMatrix::createSparseBinaryPredictor(
  const ISparseBinaryPredictorFactory& factory, const IRuleModel& ruleModel, const ILabelSpaceInfo& labelSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return ruleModel.createSparseBinaryPredictor(factory, *this, labelSpaceInfo, marginalProbabilityCalibrationModel,
                                                 jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<IScorePredictor> CContiguousFeatureMatrix::createScorePredictor(const IScorePredictorFactory& factory,
                                                                                const IRuleModel& ruleModel,
                                                                                const ILabelSpaceInfo& labelSpaceInfo,
                                                                                uint32 numLabels) const {
    return ruleModel.createScorePredictor(factory, *this, labelSpaceInfo, numLabels);
}

std::unique_ptr<IProbabilityPredictor> CContiguousFeatureMatrix::createProbabilityPredictor(
  const IProbabilityPredictorFactory& factory, const IRuleModel& ruleModel, const ILabelSpaceInfo& labelSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return ruleModel.createProbabilityPredictor(factory, *this, labelSpaceInfo, marginalProbabilityCalibrationModel,
                                                jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<ICContiguousFeatureMatrix> createCContiguousFeatureMatrix(const float32* array, uint32 numRows,
                                                                          uint32 numCols) {
    return std::make_unique<CContiguousFeatureMatrix>(array, numRows, numCols);
}
