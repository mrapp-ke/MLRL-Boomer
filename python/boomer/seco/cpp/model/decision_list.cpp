#include "decision_list.h"
#include "../../../common/cpp/model/body_empty.h"
#include "../../../common/cpp/model/body_conjunctive.h"


void DecisionListBuilder::setDefaultRule(const AbstractPrediction& prediction) {
    defaultRulePtr_ = std::make_unique<Rule>(std::make_unique<EmptyBody>(), prediction.toHead());
}

void DecisionListBuilder::addRule(const ConditionList& conditions, const AbstractPrediction& prediction) {
    modelPtr_->addRule(std::make_unique<Rule>(std::make_unique<ConjunctiveBody>(conditions), prediction.toHead()));
}

std::unique_ptr<RuleModel> DecisionListBuilder::build() {
    if (defaultRulePtr_.get() != nullptr) {
        modelPtr_->addRule(std::move(defaultRulePtr_));
    }

    return std::move(modelPtr_);
}
