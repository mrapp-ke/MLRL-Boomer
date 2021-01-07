#include "rule_list.h"
#include "../../../common/cpp/model/body_empty.h"
#include "../../../common/cpp/model/body_conjunctive.h"

using namespace boosting;


RuleListBuilder::RuleListBuilder()
    : modelPtr_(std::make_unique<RuleModel>()) {

}

void RuleListBuilder::setDefaultRule(const AbstractPrediction& prediction) {
    modelPtr_->addRule(std::make_unique<EmptyBody>(), prediction.toHead());
}

void RuleListBuilder::addRule(const ConditionList& conditions, const AbstractPrediction& prediction) {
    modelPtr_->addRule(std::make_unique<ConjunctiveBody>(conditions), prediction.toHead());
}

std::unique_ptr<RuleModel> RuleListBuilder::build() {
    return std::move(modelPtr_);
}
