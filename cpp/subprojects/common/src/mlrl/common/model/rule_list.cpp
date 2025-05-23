#include "mlrl/common/model/rule_list.hpp"

#include "mlrl/common/model/body_empty.hpp"
#include "mlrl/common/prediction/output_space_info.hpp"
#include "mlrl/common/prediction/predictor_binary.hpp"
#include "mlrl/common/prediction/predictor_probability.hpp"
#include "mlrl/common/prediction/predictor_score.hpp"

RuleList::Rule::Rule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr)
    : bodyPtr_(std::move(bodyPtr)), headPtr_(std::move(headPtr)) {}

const IBody& RuleList::Rule::getBody() const {
    return *bodyPtr_;
}

const IHead& RuleList::Rule::getHead() const {
    return *headPtr_;
}

void RuleList::Rule::visit(IBody::EmptyBodyVisitor emptyBodyVisitor,
                           IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                           IHead::CompleteHeadVisitor<uint8> completeBinaryHeadVisitor,
                           IHead::CompleteHeadVisitor<float32> complete32BitHeadVisitor,
                           IHead::CompleteHeadVisitor<float64> complete64BitHeadVisitor,
                           IHead::PartialHeadVisitor<uint8> partialBinaryHeadVisitor,
                           IHead::PartialHeadVisitor<float32> partial32BitHeadVisitor,
                           IHead::PartialHeadVisitor<float64> partial64BitHeadVisitor) const {
    bodyPtr_->visit(emptyBodyVisitor, conjunctiveBodyVisitor);
    headPtr_->visit(completeBinaryHeadVisitor, complete32BitHeadVisitor, complete64BitHeadVisitor,
                    partialBinaryHeadVisitor, partial32BitHeadVisitor, partial64BitHeadVisitor);
}

RuleList::ConstIterator::ConstIterator(bool defaultRuleTakesPrecedence, const Rule* defaultRule,
                                       std::vector<Rule>::const_iterator iterator, uint32 start, uint32 end)
    : defaultRule_(defaultRule), iterator_(iterator),
      offset_(defaultRuleTakesPrecedence && defaultRule != nullptr ? 1 : 0),
      defaultRuleIndex_(offset_ > 0 ? 0 : end - (defaultRule != nullptr ? 1 : 0)), index_(start) {}

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

RuleList::ConstIterator RuleList::ConstIterator::operator+(const uint32 difference) const {
    ConstIterator iterator(*this);
    iterator += difference;
    return iterator;
}

RuleList::ConstIterator& RuleList::ConstIterator::operator+=(const uint32 difference) {
    index_ += difference;
    return *this;
}

bool RuleList::ConstIterator::operator!=(const ConstIterator& rhs) const {
    return index_ != rhs.index_;
}

bool RuleList::ConstIterator::operator==(const ConstIterator& rhs) const {
    return index_ == rhs.index_;
}

RuleList::ConstIterator::difference_type RuleList::ConstIterator::operator-(const ConstIterator& rhs) const {
    return index_ - rhs.index_;
}

RuleList::RuleList(bool defaultRuleTakesPrecedence)
    : numUsedRules_(0), defaultRuleTakesPrecedence_(defaultRuleTakesPrecedence) {}

RuleList::const_iterator RuleList::cbegin(uint32 maxRules) const {
    uint32 numRules = maxRules > 0 ? std::min(this->getNumRules(), maxRules) : this->getNumRules();
    return ConstIterator(defaultRuleTakesPrecedence_, defaultRulePtr_.get(), ruleList_.cbegin(), 0, numRules);
}

RuleList::const_iterator RuleList::cend(uint32 maxRules) const {
    uint32 numRules = maxRules > 0 ? std::min(this->getNumRules(), maxRules) : this->getNumRules();
    return ConstIterator(defaultRuleTakesPrecedence_, defaultRulePtr_.get(), ruleList_.cbegin(), numRules, numRules);
}

RuleList::const_iterator RuleList::used_cbegin(uint32 maxRules) const {
    uint32 numRules = maxRules > 0 ? std::min(this->getNumUsedRules(), maxRules) : this->getNumUsedRules();
    return ConstIterator(defaultRuleTakesPrecedence_, defaultRulePtr_.get(), ruleList_.cbegin(), 0, numRules);
}

RuleList::const_iterator RuleList::used_cend(uint32 maxRules) const {
    uint32 numRules = maxRules > 0 ? std::min(this->getNumUsedRules(), maxRules) : this->getNumUsedRules();
    return ConstIterator(defaultRuleTakesPrecedence_, defaultRulePtr_.get(), ruleList_.cbegin(), numRules, numRules);
}

uint32 RuleList::getNumRules() const {
    uint32 numRules = static_cast<uint32>(ruleList_.size());

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
                     IHead::CompleteHeadVisitor<uint8> completeBinaryHeadVisitor,
                     IHead::CompleteHeadVisitor<float32> complete32BitHeadVisitor,
                     IHead::CompleteHeadVisitor<float64> complete64BitHeadVisitor,
                     IHead::PartialHeadVisitor<uint8> partialBinaryHeadVisitor,
                     IHead::PartialHeadVisitor<float32> partial32BitHeadVisitor,
                     IHead::PartialHeadVisitor<float64> partial64BitHeadVisitor) const {
    for (auto it = this->cbegin(); it != this->cend(); it++) {
        const Rule& rule = *it;
        rule.visit(emptyBodyVisitor, conjunctiveBodyVisitor, completeBinaryHeadVisitor, complete32BitHeadVisitor,
                   complete64BitHeadVisitor, partialBinaryHeadVisitor, partial32BitHeadVisitor,
                   partial64BitHeadVisitor);
    }
}

void RuleList::visitUsed(IBody::EmptyBodyVisitor emptyBodyVisitor, IBody::ConjunctiveBodyVisitor conjunctiveBodyVisitor,
                         IHead::CompleteHeadVisitor<uint8> completeBinaryHeadVisitor,
                         IHead::CompleteHeadVisitor<float32> complete32BitHeadVisitor,
                         IHead::CompleteHeadVisitor<float64> complete64BitHeadVisitor,
                         IHead::PartialHeadVisitor<uint8> partialBinaryHeadVisitor,
                         IHead::PartialHeadVisitor<float32> partial32BitHeadVisitor,
                         IHead::PartialHeadVisitor<float64> partial64BitHeadVisitor) const {
    for (auto it = this->used_cbegin(); it != this->used_cend(); it++) {
        const Rule& rule = *it;
        rule.visit(emptyBodyVisitor, conjunctiveBodyVisitor, completeBinaryHeadVisitor, complete32BitHeadVisitor,
                   complete64BitHeadVisitor, partialBinaryHeadVisitor, partial32BitHeadVisitor,
                   partial64BitHeadVisitor);
    }
}

std::unique_ptr<IBinaryPredictor> RuleList::createBinaryPredictor(
  const IBinaryPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
  const IOutputSpaceInfo& outputSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return outputSpaceInfo.createBinaryPredictor(factory, featureMatrix, *this, marginalProbabilityCalibrationModel,
                                                 jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<IBinaryPredictor> RuleList::createBinaryPredictor(
  const IBinaryPredictorFactory& factory, const CsrView<const float32>& featureMatrix,
  const IOutputSpaceInfo& outputSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return outputSpaceInfo.createBinaryPredictor(factory, featureMatrix, *this, marginalProbabilityCalibrationModel,
                                                 jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<ISparseBinaryPredictor> RuleList::createSparseBinaryPredictor(
  const ISparseBinaryPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
  const IOutputSpaceInfo& outputSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return outputSpaceInfo.createSparseBinaryPredictor(
      factory, featureMatrix, *this, marginalProbabilityCalibrationModel, jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<ISparseBinaryPredictor> RuleList::createSparseBinaryPredictor(
  const ISparseBinaryPredictorFactory& factory, const CsrView<const float32>& featureMatrix,
  const IOutputSpaceInfo& outputSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return outputSpaceInfo.createSparseBinaryPredictor(
      factory, featureMatrix, *this, marginalProbabilityCalibrationModel, jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<IScorePredictor> RuleList::createScorePredictor(const IScorePredictorFactory& factory,
                                                                const CContiguousView<const float32>& featureMatrix,
                                                                const IOutputSpaceInfo& outputSpaceInfo,
                                                                uint32 numOutputs) const {
    return outputSpaceInfo.createScorePredictor(factory, featureMatrix, *this, numOutputs);
}

std::unique_ptr<IScorePredictor> RuleList::createScorePredictor(const IScorePredictorFactory& factory,
                                                                const CsrView<const float32>& featureMatrix,
                                                                const IOutputSpaceInfo& outputSpaceInfo,
                                                                uint32 numOutputs) const {
    return outputSpaceInfo.createScorePredictor(factory, featureMatrix, *this, numOutputs);
}

std::unique_ptr<IProbabilityPredictor> RuleList::createProbabilityPredictor(
  const IProbabilityPredictorFactory& factory, const CContiguousView<const float32>& featureMatrix,
  const IOutputSpaceInfo& outputSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return outputSpaceInfo.createProbabilityPredictor(
      factory, featureMatrix, *this, marginalProbabilityCalibrationModel, jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<IProbabilityPredictor> RuleList::createProbabilityPredictor(
  const IProbabilityPredictorFactory& factory, const CsrView<const float32>& featureMatrix,
  const IOutputSpaceInfo& outputSpaceInfo,
  const IMarginalProbabilityCalibrationModel& marginalProbabilityCalibrationModel,
  const IJointProbabilityCalibrationModel& jointProbabilityCalibrationModel, uint32 numLabels) const {
    return outputSpaceInfo.createProbabilityPredictor(
      factory, featureMatrix, *this, marginalProbabilityCalibrationModel, jointProbabilityCalibrationModel, numLabels);
}

std::unique_ptr<IRuleList> createRuleList(bool defaultRuleTakesPrecedence) {
    return std::make_unique<RuleList>(defaultRuleTakesPrecedence);
}
