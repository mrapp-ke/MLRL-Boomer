#include "rule_model.h"


RuleModel::const_iterator RuleModel::cbegin() const {
    return list_.cbegin();
}

RuleModel::const_iterator RuleModel::cend() const {
    return list_.cend();
}

void RuleModel::addRule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr) {
    list_.emplace_back(std::move(bodyPtr), std::move(headPtr));
}
