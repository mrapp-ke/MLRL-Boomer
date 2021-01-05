#include "rule_model.h"


void RuleModel::addRule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr) {
    list_.emplace_back(std::move(bodyPtr), std::move(headPtr));
}
