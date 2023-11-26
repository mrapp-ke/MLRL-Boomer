#include "mlrl/common/input/feature_matrix_csr.hpp"

#include "mlrl/common/prediction/predictor_binary.hpp"
#include "mlrl/common/prediction/predictor_probability.hpp"
#include "mlrl/common/prediction/predictor_score.hpp"

CsrFeatureMatrix::CsrFeatureMatrix(const float32* values, uint32* indices, uint32* indptr, uint32 numRows,
                                   uint32 numCols)
    : CsrView<const float32>(values, indices, indptr, numRows, numCols) {}

bool CsrFeatureMatrix::isSparse() const {
    return true;
}

uint32 CsrFeatureMatrix::getNumExamples() const {
    return Matrix::numRows;
}

uint32 CsrFeatureMatrix::getNumFeatures() const {
    return Matrix::numCols;
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

std::unique_ptr<ICsrFeatureMatrix> createCsrFeatureMatrix(const float32* values, uint32* indices, uint32* indptr,
                                                          uint32 numRows, uint32 numCols) {
    return std::make_unique<CsrFeatureMatrix>(values, indices, indptr, numRows, numCols);
}
