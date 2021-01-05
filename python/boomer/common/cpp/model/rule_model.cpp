#include "rule_model.h"


void RuleModel::addRule(std::unique_ptr<Rule> rulePtr) {
    list_.push_back(std::move(rulePtr));
}
