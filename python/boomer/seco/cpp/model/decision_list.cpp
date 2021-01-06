#include "decision_list.h"
#include "../../../common/cpp/model/body_empty.h"
#include "../../../common/cpp/model/body_conjunctive.h"

using namespace seco;


void DecisionListBuilder::setDefaultRule(const AbstractPrediction& prediction) {
    defaultHeadPtr_ = prediction.toHead();
}

void DecisionListBuilder::addRule(const ConditionList& conditions, const AbstractPrediction& prediction) {
    modelPtr_->addRule(std::make_unique<ConjunctiveBody>(conditions), prediction.toHead());
}

std::unique_ptr<RuleModel> DecisionListBuilder::build() {
    if (defaultHeadPtr_.get() != nullptr) {
        modelPtr_->addRule(std::make_unique<EmptyBody>(), std::move(defaultHeadPtr_));
    }

    return std::move(modelPtr_);
}
