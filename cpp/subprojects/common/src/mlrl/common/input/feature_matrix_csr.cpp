#include "mlrl/common/input/feature_matrix_csr.hpp"

#include "mlrl/common/prediction/predictor_binary.hpp"
#include "mlrl/common/prediction/predictor_probability.hpp"
#include "mlrl/common/prediction/predictor_score.hpp"

CsrFeatureMatrix::CsrFeatureMatrix(uint32 numRows, uint32 numCols, const float32* data, uint32* colIndices,
                                   uint32* indptr)
    : CsrConstView<const float32>(numRows, numCols, data, colIndices, indptr) {}

bool CsrFeatureMatrix::isSparse() const {
    return true;
}

std::unique_ptr<IBinaryPredictor> CsrFeatureMatrix::createBinaryPredictor(
  const IBinaryPredictorFactory& factory, const IRuleModel& ruleModel, const ILabelSpaceInfo& labelSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return ruleModel.createBinaryPredictor(factory, *this, labelSpaceInfo, marginalProbabilityCalibrationModel,
                                           jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<ISparseBinaryPredictor> CsrFeatureMatrix::createSparseBinaryPredictor(
  const ISparseBinaryPredictorFactory& factory, const IRuleModel& ruleModel, const ILabelSpaceInfo& labelSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return ruleModel.createSparseBinaryPredictor(factory, *this, labelSpaceInfo, marginalProbabilityCalibrationModel,
                                                 jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<IScorePredictor> CsrFeatureMatrix::createScorePredictor(const IScorePredictorFactory& factory,
                                                                        const IRuleModel& ruleModel,
                                                                        const ILabelSpaceInfo& labelSpaceInfo,
                                                                        uint32 numLabels) const {
    return ruleModel.createScorePredictor(factory, *this, labelSpaceInfo, numLabels);
}

std::unique_ptr<IProbabilityPredictor> CsrFeatureMatrix::createProbabilityPredictor(
  const IProbabilityPredictorFactory& factory, const IRuleModel& ruleModel, const ILabelSpaceInfo& labelSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return ruleModel.createProbabilityPredictor(factory, *this, labelSpaceInfo, marginalProbabilityCalibrationModel,
                                                jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<ICsrFeatureMatrix> createCsrFeatureMatrix(uint32 numRows, uint32 numCols, const float32* data,
                                                          uint32* colIndices, uint32* indptr) {
    return std::make_unique<CsrFeatureMatrix>(numRows, numCols, data, colIndices, indptr);
}