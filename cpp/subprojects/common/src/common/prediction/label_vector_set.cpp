#include "common/prediction/label_vector_set.hpp"

#include "common/input/feature_matrix_c_contiguous.hpp"
#include "common/input/feature_matrix_csr.hpp"
#include "common/model/rule_list.hpp"
#include "common/prediction/predictor_binary.hpp"
#include "common/prediction/predictor_probability.hpp"
#include "common/prediction/predictor_score.hpp"
#include "common/prediction/probability_calibration.hpp"

LabelVectorSet::const_iterator LabelVectorSet::cbegin() const {
    return labelVectors_.cbegin();
}

LabelVectorSet::const_iterator LabelVectorSet::cend() const {
    return labelVectors_.cend();
}

uint32 LabelVectorSet::getNumLabelVectors() const {
    return (uint32) labelVectors_.size();
}

void LabelVectorSet::addLabelVector(std::unique_ptr<LabelVector> labelVectorPtr) {
    ++labelVectors_[std::move(labelVectorPtr)];
}

void LabelVectorSet::visit(LabelVectorVisitor visitor) const {
    for (auto it = labelVectors_.cbegin(); it != labelVectors_.cend(); it++) {
        const auto& entry = *it;
        const std::unique_ptr<LabelVector>& labelVectorPtr = entry.first;
        visitor(*labelVectorPtr);
    }
}

std::unique_ptr<IBinaryPredictor> LabelVectorSet::createBinaryPredictor(
  const IBinaryPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix, const RuleList& model,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel, uint32 numLabels) const {
    return factory.create(featureMatrix, model, this, marginalProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<IBinaryPredictor> LabelVectorSet::createBinaryPredictor(
  const IBinaryPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix, const RuleList& model,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel, uint32 numLabels) const {
    return factory.create(featureMatrix, model, this, marginalProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<ISparseBinaryPredictor> LabelVectorSet::createSparseBinaryPredictor(
  const ISparseBinaryPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix, const RuleList& model,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel, uint32 numLabels) const {
    return factory.create(featureMatrix, model, this, marginalProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<ISparseBinaryPredictor> LabelVectorSet::createSparseBinaryPredictor(
  const ISparseBinaryPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix, const RuleList& model,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel, uint32 numLabels) const {
    return factory.create(featureMatrix, model, this, marginalProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<IScorePredictor> LabelVectorSet::createScorePredictor(
  const IScorePredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix, const RuleList& model,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel, uint32 numLabels) const {
    return factory.create(featureMatrix, model, this, marginalProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<IScorePredictor> LabelVectorSet::createScorePredictor(
  const IScorePredictorFactory& factory, const CsrFeatureMatrix& featureMatrix, const RuleList& model,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel, uint32 numLabels) const {
    return factory.create(featureMatrix, model, this, marginalProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<IProbabilityPredictor> LabelVectorSet::createProbabilityPredictor(
  const IProbabilityPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix, const RuleList& model,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel, uint32 numLabels) const {
    return factory.create(featureMatrix, model, this, marginalProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<IProbabilityPredictor> LabelVectorSet::createProbabilityPredictor(
  const IProbabilityPredictorFactory& factory, const CsrFeatureMatrix& featureMatrix, const RuleList& model,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel, uint32 numLabels) const {
    return factory.create(featureMatrix, model, this, marginalProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<ILabelVectorSet> createLabelVectorSet() {
    return std::make_unique<LabelVectorSet>();
}

std::unique_ptr<ILabelVectorSet> createLabelVectorSet(const IRowWiseLabelMatrix& labelMatrix) {
    std::unique_ptr<LabelVectorSet> labelVectorSetPtr = std::make_unique<LabelVectorSet>();
    uint32 numRows = labelMatrix.getNumRows();

    for (uint32 i = 0; i < numRows; i++) {
        labelVectorSetPtr->addLabelVector(labelMatrix.createLabelVector(i));
    }

    return labelVectorSetPtr;
}
