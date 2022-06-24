#include "common/post_optimization/model_builder_intermediate.hpp"


IntermediateModelBuilder::IntermediateModelBuilder(std::unique_ptr<IModelBuilder> modelBuilderPtr)
    : modelBuilderPtr_(std::move(modelBuilderPtr)) {

}

void IntermediateModelBuilder::setDefaultRule(std::unique_ptr<AbstractEvaluatedPrediction>& predictionPtr) {
    defaultPredictionPtr_ = std::move(predictionPtr);
}

void IntermediateModelBuilder::addRule(std::unique_ptr<ConditionList>& conditionListPtr,
                                       std::unique_ptr<AbstractEvaluatedPrediction>& predictionPtr) {
    intermediateRuleList_.emplace_back(std::move(conditionListPtr), std::move(predictionPtr));
}

std::unique_ptr<IRuleModel> IntermediateModelBuilder::buildModel(uint32 numUsedRules) {
    if (defaultPredictionPtr_) {
        modelBuilderPtr_->setDefaultRule(defaultPredictionPtr_);
        defaultPredictionPtr_.release();
    }

    for (auto it = intermediateRuleList_.begin(); it != intermediateRuleList_.end(); it++) {
        IntermediateRule& intermediateRule = *it;
        modelBuilderPtr_->addRule(intermediateRule.first, intermediateRule.second);
    }

    intermediateRuleList_.clear();
    return modelBuilderPtr_->buildModel(numUsedRules);
}
