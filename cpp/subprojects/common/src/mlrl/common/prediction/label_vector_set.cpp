#include "mlrl/common/prediction/label_vector_set.hpp"

#include "mlrl/common/model/rule_list.hpp"
#include "mlrl/common/prediction/predictor_binary.hpp"
#include "mlrl/common/prediction/predictor_probability.hpp"
#include "mlrl/common/prediction/predictor_score.hpp"
#include "mlrl/common/prediction/probability_calibration_joint.hpp"
#include "mlrl/common/prediction/probability_calibration_marginal.hpp"

#include <unordered_map>

LabelVectorSet::LabelVectorSet() {}

LabelVectorSet::LabelVectorSet(const IRowWiseLabelMatrix& labelMatrix) {
    typedef typename LabelVector::view_type Key;
    typedef typename LabelVector::view_type::Hash Hash;
    typedef typename LabelVector::view_type::Equal Equal;
    std::unordered_map<std::reference_wrapper<Key>, uint32, Hash, Equal> map;
    uint32 numExamples = labelMatrix.getNumExamples();

    for (uint32 i = 0; i < numExamples; i++) {
        std::unique_ptr<LabelVector> labelVectorPtr = labelMatrix.createLabelVector(i);
        auto it = map.find(labelVectorPtr->getView());

        if (it == map.end()) {
            labelVectors_.push_back(std::move(labelVectorPtr));
            map.emplace(labelVectors_.back()->getView(), static_cast<uint32>(frequencies_.size()));
            frequencies_.emplace_back(1);
        } else {
            uint32 index = (*it).second;
            frequencies_[index] += 1;
        }
    }
}

LabelVectorSet::const_iterator LabelVectorSet::cbegin() const {
    return labelVectors_.cbegin();
}

LabelVectorSet::const_iterator LabelVectorSet::cend() const {
    return labelVectors_.cend();
}

LabelVectorSet::frequency_const_iterator LabelVectorSet::frequencies_cbegin() const {
    return frequencies_.cbegin();
}

LabelVectorSet::frequency_const_iterator LabelVectorSet::frequencies_cend() const {
    return frequencies_.cend();
}

uint32 LabelVectorSet::getNumLabelVectors() const {
    return static_cast<uint32>(labelVectors_.size());
}

void LabelVectorSet::addLabelVector(std::unique_ptr<LabelVector> labelVectorPtr, uint32 frequency) {
    labelVectors_.push_back(std::move(labelVectorPtr));
    frequencies_.emplace_back(frequency);
}

void LabelVectorSet::visit(LabelVectorVisitor visitor) const {
    uint32 numLabelVectors = this->getNumLabelVectors();

    for (uint32 i = 0; i < numLabelVectors; i++) {
        const std::unique_ptr<LabelVector>& labelVectorPtr = labelVectors_[i];
        uint32 frequency = frequencies_[i];
        visitor(*labelVectorPtr, frequency);
    }
}

std::unique_ptr<IJointProbabilityCalibrator> LabelVectorSet::createJointProbabilityCalibrator(
  const IJointProbabilityCalibratorFactory& factory,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel) const {
    return factory.create(marginalProbabilityCalibrationModel, this);
}

std::unique_ptr<IBinaryPredictor> LabelVectorSet::createBinaryPredictor(
  const IBinaryPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix, const RuleList& model,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return factory.create(featureMatrix, model, this, marginalProbabilityCalibrationModel,
                          jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<IBinaryPredictor> LabelVectorSet::createBinaryPredictor(
  const IBinaryPredictorFactory& factory, const CsrView<const float32>& featureMatrix, const RuleList& model,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return factory.create(featureMatrix, model, this, marginalProbabilityCalibrationModel,
                          jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<ISparseBinaryPredictor> LabelVectorSet::createSparseBinaryPredictor(
  const ISparseBinaryPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
  const RuleList& model, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return factory.create(featureMatrix, model, this, marginalProbabilityCalibrationModel,
                          jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<ISparseBinaryPredictor> LabelVectorSet::createSparseBinaryPredictor(
  const ISparseBinaryPredictorFactory& factory, const CsrView<const float32>& featureMatrix, const RuleList& model,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return factory.create(featureMatrix, model, this, marginalProbabilityCalibrationModel,
                          jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<IScorePredictor> LabelVectorSet::createScorePredictor(
  const IScorePredictorFactory& factory, const CContiguousView<const float32>& featureMatrix, const RuleList& model,
  uint32 numOutputs) const {
    return factory.create(featureMatrix, model, this, numOutputs);
}

std::unique_ptr<IScorePredictor> LabelVectorSet::createScorePredictor(const IScorePredictorFactory& factory,
                                                                      const CsrView<const float32>& featureMatrix,
                                                                      const RuleList& model, uint32 numOutputs) const {
    return factory.create(featureMatrix, model, this, numOutputs);
}

std::unique_ptr<IProbabilityPredictor> LabelVectorSet::createProbabilityPredictor(
  const IProbabilityPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
  const RuleList& model, const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return factory.create(featureMatrix, model, this, marginalProbabilityCalibrationModel,
                          jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<IProbabilityPredictor> LabelVectorSet::createProbabilityPredictor(
  const IProbabilityPredictorFactory& factory, const CsrView<const float32>& featureMatrix, const RuleList& model,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return factory.create(featureMatrix, model, this, marginalProbabilityCalibrationModel,
                          jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<ILabelVectorSet> createLabelVectorSet() {
    return std::make_unique<LabelVectorSet>();
}
