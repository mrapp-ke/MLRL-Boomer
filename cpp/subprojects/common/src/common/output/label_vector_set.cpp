#include "common/output/label_vector_set.hpp"
#include "common/output/predictor_classification.hpp"
#include "common/output/predictor_regression.hpp"
#include "common/output/predictor_probability.hpp"
#include "common/model/rule_list.hpp"


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

std::unique_ptr<IClassificationPredictor> LabelVectorSet::createClassificationPredictor(
        const IClassificationPredictorFactory& factory, const RuleList& model) const {
    return factory.create(model, this);
}

std::unique_ptr<IOldRegressionPredictor> LabelVectorSet::createRegressionPredictor(
        const IRegressionPredictorFactory& factory, const RuleList& model) const {
    return factory.create(model, this);
}

std::unique_ptr<IOldProbabilityPredictor> LabelVectorSet::createProbabilityPredictor(
        const IProbabilityPredictorFactory& factory, const RuleList& model) const {
    return factory.create(model, this);
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
