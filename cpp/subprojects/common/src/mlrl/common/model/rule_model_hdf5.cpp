#include "mlrl/common/model/rule_model_hdf5.hpp"

#include "mlrl/common/prediction/predictor_binary.hpp"
#include "mlrl/common/prediction/predictor_probability.hpp"
#include "mlrl/common/prediction/predictor_score.hpp"

Hdf5RuleModel::Hdf5RuleModel(bool defaultRuleTakesPrecedence) {
    // TODO
}

uint32 Hdf5RuleModel::getNumRules() const {
    // TODO
    return 0;
}

uint32 Hdf5RuleModel::getNumUsedRules() const {
    // TODO
    return 0;
}

void Hdf5RuleModel::setNumUsedRules(uint32 numUsedRules) {
    // TODO
}

std::unique_ptr<IBinaryPredictor> Hdf5RuleModel::createBinaryPredictor(
  const IBinaryPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
  const IOutputSpaceInfo& outputSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    // TODO
    return nullptr;
}

std::unique_ptr<IBinaryPredictor> Hdf5RuleModel::createBinaryPredictor(
  const IBinaryPredictorFactory& factory, const CsrView<const float32>& featureMatrix,
  const IOutputSpaceInfo& outputSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    // TODO
    return nullptr;
}

std::unique_ptr<ISparseBinaryPredictor> Hdf5RuleModel::createSparseBinaryPredictor(
  const ISparseBinaryPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
  const IOutputSpaceInfo& outputSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    // TODO
    return nullptr;
}

std::unique_ptr<ISparseBinaryPredictor> Hdf5RuleModel::createSparseBinaryPredictor(
  const ISparseBinaryPredictorFactory& factory, const CsrView<const float32>& featureMatrix,
  const IOutputSpaceInfo& outputSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    // TODO
    return nullptr;
}

std::unique_ptr<IScorePredictor> Hdf5RuleModel::createScorePredictor(
  const IScorePredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
  const IOutputSpaceInfo& outputSpaceInfo, uint32 numOutputs) const {
    // TODO
    return nullptr;
}

std::unique_ptr<IScorePredictor> Hdf5RuleModel::createScorePredictor(const IScorePredictorFactory& factory,
                                                                     const CsrView<const float32>& featureMatrix,
                                                                     const IOutputSpaceInfo& outputSpaceInfo,
                                                                     uint32 numOutputs) const {
    // TODO
    return nullptr;
}

std::unique_ptr<IProbabilityPredictor> Hdf5RuleModel::createProbabilityPredictor(
  const IProbabilityPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
  const IOutputSpaceInfo& outputSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    // TODO
    return nullptr;
}

std::unique_ptr<IProbabilityPredictor> Hdf5RuleModel::createProbabilityPredictor(
  const IProbabilityPredictorFactory& factory, const CsrView<const float32>& featureMatrix,
  const IOutputSpaceInfo& outputSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    // TODO
    return nullptr;
}
