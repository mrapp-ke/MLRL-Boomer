#include "common/model/rule_model.hpp"


RuleModel::const_iterator RuleModel::cbegin() const {
    return list_.cbegin();
}

RuleModel::const_iterator RuleModel::cend() const {
    return list_.cend();
}

uint32 RuleModel::getNumRules() const {
    return (uint32) list_.size();
}

void RuleModel::addRule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr) {
    list_.emplace_back(std::move(bodyPtr), std::move(headPtr));
}

void RuleModel::visit(IBody::EmptyBodyVisitor emptyBodyVisitor, IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                      IHead::FullHeadVisitor fullHeadVisitor, IHead::PartialHeadVisitor partialHeadVisitor) const {
    for (auto it = list_.cbegin(); it != list_.cend(); it++) {
        const Rule& rule = *it;
        rule.visit(emptyBodyVisitor, conjunctiveBodyVisitor, fullHeadVisitor, partialHeadVisitor);
    }
}
