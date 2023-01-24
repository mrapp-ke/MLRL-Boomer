#include "common/model/rule_list.hpp"
#include "common/model/body_empty.hpp"
#include "common/output/predictor_classification.hpp"
#include "common/output/predictor_regression.hpp"
#include "common/output/predictor_probability.hpp"
#include "common/prediction/label_space_info.hpp"
#include "common/prediction/predictor_label.hpp"
#include "common/prediction/predictor_probability.hpp"
#include "common/prediction/predictor_score.hpp"


RuleList::Rule::Rule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr)
    : bodyPtr_(std::move(bodyPtr)), headPtr_(std::move(headPtr)) {

}

const IBody& RuleList::Rule::getBody() const {
    return *bodyPtr_;
}

const IHead& RuleList::Rule::getHead() const {
    return *headPtr_;
}

void RuleList::Rule::visit(IBody::EmptyBodyVisitor emptyBodyVisitor,
                           IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                           IHead::CompleteHeadVisitor completeHeadVisitor,
                           IHead::PartialHeadVisitor partialHeadVisitor) const {
    bodyPtr_->visit(emptyBodyVisitor, conjunctiveBodyVisitor);
    headPtr_->visit(completeHeadVisitor, partialHeadVisitor);
}

RuleList::ConstIterator::ConstIterator(bool defaultRuleTakesPrecedence, const Rule* defaultRule,
                                       std::vector<Rule>::const_iterator iterator, uint32 start, uint32 end)
    : defaultRule_(defaultRule), iterator_(iterator),
      offset_(defaultRuleTakesPrecedence && defaultRule != nullptr ? 1 : 0),
      defaultRuleIndex_(offset_ > 0 ? 0 : end - (defaultRule != nullptr ? 1 : 0)), index_(start) {

}

RuleList::ConstIterator::reference RuleList::ConstIterator::operator*() const {
    uint32 index = index_;

    if (index == defaultRuleIndex_) {
        return *defaultRule_;
    } else {
        return iterator_[index - offset_];
    }
}

RuleList::ConstIterator& RuleList::ConstIterator::operator++() {
    ++index_;
    return *this;
}

RuleList::ConstIterator& RuleList::ConstIterator::operator++(int n) {
    index_++;
    return *this;
}

bool RuleList::ConstIterator::operator!=(const ConstIterator& rhs) const {
    return index_ != rhs.index_;
}

bool RuleList::ConstIterator::operator==(const ConstIterator& rhs) const {
    return index_ == rhs.index_;
}

RuleList::RuleList(bool defaultRuleTakesPrecedence)
    : numUsedRules_(0), defaultRuleTakesPrecedence_(defaultRuleTakesPrecedence) {

}

RuleList::const_iterator RuleList::cbegin() const {
    return ConstIterator(defaultRuleTakesPrecedence_, defaultRulePtr_.get(), ruleList_.cbegin(), 0,
                         this->getNumRules());
}

RuleList::const_iterator RuleList::cend() const {
    uint32 numRules = this->getNumRules();
    return ConstIterator(defaultRuleTakesPrecedence_, defaultRulePtr_.get(), ruleList_.cbegin(), numRules, numRules);
}

RuleList::const_iterator RuleList::used_cbegin() const {
    return ConstIterator(defaultRuleTakesPrecedence_, defaultRulePtr_.get(), ruleList_.cbegin(), 0,
                         this->getNumUsedRules());
}

RuleList::const_iterator RuleList::used_cend() const {
    uint32 numUsedRules = this->getNumUsedRules();
    return ConstIterator(defaultRuleTakesPrecedence_, defaultRulePtr_.get(), ruleList_.cbegin(), numUsedRules,
                         numUsedRules);
}

uint32 RuleList::getNumRules() const {
    uint32 numRules = (uint32) ruleList_.size();

    if (this->containsDefaultRule()) {
        numRules++;
    }

    return numRules;
}

uint32 RuleList::getNumUsedRules() const {
    return numUsedRules_ > 0 ? numUsedRules_ : this->getNumRules();
}

void RuleList::setNumUsedRules(uint32 numUsedRules) {
    numUsedRules_ = numUsedRules;
}

void RuleList::addDefaultRule(std::unique_ptr<IHead> headPtr) {
    defaultRulePtr_ = std::make_unique<Rule>(std::make_unique<EmptyBody>(), std::move(headPtr));
}

void RuleList::addRule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr) {
    ruleList_.emplace_back(std::move(bodyPtr), std::move(headPtr));
}

bool RuleList::containsDefaultRule() const {
    return defaultRulePtr_ != nullptr;
}

bool RuleList::isDefaultRuleTakingPrecedence() const {
    return defaultRuleTakesPrecedence_;
}

void RuleList::visit(IBody::EmptyBodyVisitor emptyBodyVisitor, IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                      IHead::CompleteHeadVisitor completeHeadVisitor,
                      IHead::PartialHeadVisitor partialHeadVisitor) const {
    for (auto it = this->cbegin(); it != this->cend(); it++) {
        const Rule& rule = *it;
        rule.visit(emptyBodyVisitor, conjunctiveBodyVisitor, completeHeadVisitor, partialHeadVisitor);
    }
}

void RuleList::visitUsed(IBody::EmptyBodyVisitor emptyBodyVisitor,
                          IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                          IHead::CompleteHeadVisitor completeHeadVisitor,
                          IHead::PartialHeadVisitor partialHeadVisitor) const {
    for (auto it = this->used_cbegin(); it != this->used_cend(); it++) {
        const Rule& rule = *it;
        rule.visit(emptyBodyVisitor, conjunctiveBodyVisitor, completeHeadVisitor, partialHeadVisitor);
    }
}

std::unique_ptr<ILabelPredictor> RuleList::createLabelPredictor(const ILabelPredictorFactory& factory,
                                                                const CContiguousFeatureMatrix& featureMatrix,
                                                                const ILabelSpaceInfo& labelSpaceInfo,
                                                                uint32 numLabels) const {
    return labelSpaceInfo.createLabelPredictor(factory, featureMatrix, *this, numLabels);
}

std::unique_ptr<ILabelPredictor> RuleList::createLabelPredictor(const ILabelPredictorFactory& factory,
                                                                const CsrFeatureMatrix& featureMatrix,
                                                                const ILabelSpaceInfo& labelSpaceInfo,
                                                                uint32 numLabels) const {
    return labelSpaceInfo.createLabelPredictor(factory, featureMatrix, *this, numLabels);
}

std::unique_ptr<ISparseLabelPredictor> RuleList::createSparseLabelPredictor(const ISparseLabelPredictorFactory& factory,
                                                                          const CContiguousFeatureMatrix& featureMatrix,
                                                                          const ILabelSpaceInfo& labelSpaceInfo,
                                                                          uint32 numLabels) const {
    return labelSpaceInfo.createSparseLabelPredictor(factory, featureMatrix, *this, numLabels);
}

std::unique_ptr<ISparseLabelPredictor> RuleList::createSparseLabelPredictor(const ISparseLabelPredictorFactory& factory,
                                                                          const CsrFeatureMatrix& featureMatrix,
                                                                          const ILabelSpaceInfo& labelSpaceInfo,
                                                                          uint32 numLabels) const {
    return labelSpaceInfo.createSparseLabelPredictor(factory, featureMatrix, *this, numLabels);
}

// TODO Remove
std::unique_ptr<IClassificationPredictor> RuleList::createClassificationPredictor(
        const IClassificationPredictorFactory& factory, const ILabelSpaceInfo& labelSpaceInfo) const {
    return nullptr;
}

std::unique_ptr<IScorePredictor> RuleList::createScorePredictor(const IScorePredictorFactory& factory,
                                                                const CContiguousFeatureMatrix& featureMatrix,
                                                                const ILabelSpaceInfo& labelSpaceInfo,
                                                                uint32 numLabels) const {
    return labelSpaceInfo.createScorePredictor(factory, featureMatrix, *this, numLabels);
}

std::unique_ptr<IScorePredictor> RuleList::createScorePredictor(const IScorePredictorFactory& factory,
                                                                const CsrFeatureMatrix& featureMatrix,
                                                                const ILabelSpaceInfo& labelSpaceInfo,
                                                                uint32 numLabels) const {
    return labelSpaceInfo.createScorePredictor(factory, featureMatrix, *this, numLabels);
}

// TODO Remove
std::unique_ptr<IOldRegressionPredictor> RuleList::createRegressionPredictor(
        const IRegressionPredictorFactory& factory, const ILabelSpaceInfo& labelSpaceInfo) const {
    return nullptr;
}

std::unique_ptr<IProbabilityPredictor> RuleList::createProbabilityPredictor(
        const IProbabilityPredictorFactory& factory, const CContiguousFeatureMatrix& featureMatrix,
        const ILabelSpaceInfo& labelSpaceInfo, uint32 numLabels) const {
    return labelSpaceInfo.createProbabilityPredictor(factory, featureMatrix, *this, numLabels);
}

std::unique_ptr<IProbabilityPredictor> RuleList::createProbabilityPredictor(const IProbabilityPredictorFactory& factory,
                                                                            const CsrFeatureMatrix& featureMatrix,
                                                                            const ILabelSpaceInfo& labelSpaceInfo,
                                                                            uint32 numLabels) const {
    return labelSpaceInfo.createProbabilityPredictor(factory, featureMatrix, *this, numLabels);
}

// TODO Remove
std::unique_ptr<IOldProbabilityPredictor> RuleList::createProbabilityPredictor(
        const IOldProbabilityPredictorFactory& factory, const ILabelSpaceInfo& labelSpaceInfo) const {
    return nullptr;
}

std::unique_ptr<IRuleList> createRuleList(bool defaultRuleTakesPrecedence) {
    return std::make_unique<RuleList>(defaultRuleTakesPrecedence);
}
