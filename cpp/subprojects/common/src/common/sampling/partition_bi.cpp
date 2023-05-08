#include "common/sampling/partition_bi.hpp"

#include "common/rule_refinement/prediction.hpp"
#include "common/sampling/instance_sampling.hpp"
#include "common/stopping/stopping_criterion.hpp"
#include "common/thresholds/thresholds_subset.hpp"

#include <algorithm>

BiPartition::BiPartition(uint32 numFirst, uint32 numSecond)
    : vector_(DenseVector<uint32>(numFirst + numSecond)), numFirst_(numFirst), firstSorted_(false),
      secondSorted_(false) {}

BiPartition::iterator BiPartition::first_begin() {
    return vector_.begin();
}

BiPartition::iterator BiPartition::first_end() {
    return &vector_.begin()[numFirst_];
}

BiPartition::const_iterator BiPartition::first_cbegin() const {
    return vector_.cbegin();
}

BiPartition::const_iterator BiPartition::first_cend() const {
    return &vector_.cbegin()[numFirst_];
}

BiPartition::iterator BiPartition::second_begin() {
    return &vector_.begin()[numFirst_];
}

BiPartition::iterator BiPartition::second_end() {
    return vector_.end();
}

BiPartition::const_iterator BiPartition::second_cbegin() const {
    return &vector_.cbegin()[numFirst_];
}

BiPartition::const_iterator BiPartition::second_cend() const {
    return vector_.cend();
}

uint32 BiPartition::getNumFirst() const {
    return numFirst_;
}

uint32 BiPartition::getNumSecond() const {
    return vector_.getNumElements() - numFirst_;
}

uint32 BiPartition::getNumElements() const {
    return vector_.getNumElements();
}

void BiPartition::sortFirst() {
    if (!firstSorted_) {
        std::sort(this->first_begin(), this->first_end(), std::less<uint32>());
        firstSorted_ = true;
    }
}

void BiPartition::sortSecond() {
    if (!secondSorted_) {
        std::sort(this->second_begin(), this->second_end(), std::less<uint32>());
        secondSorted_ = true;
    }
}

std::unique_ptr<IStoppingCriterion> BiPartition::createStoppingCriterion(const IStoppingCriterionFactory& factory) {
    return factory.create(*this);
}

std::unique_ptr<IInstanceSampling> BiPartition::createInstanceSampling(const IInstanceSamplingFactory& factory,
                                                                       const IRowWiseLabelMatrix& labelMatrix,
                                                                       IStatistics& statistics) {
    return labelMatrix.createInstanceSampling(factory, *this, statistics);
}

Quality BiPartition::evaluateOutOfSample(const IThresholdsSubset& thresholdsSubset, const ICoverageState& coverageState,
                                         const AbstractPrediction& head) {
    return coverageState.evaluateOutOfSample(thresholdsSubset, *this, head);
}

void BiPartition::recalculatePrediction(const IThresholdsSubset& thresholdsSubset, const ICoverageState& coverageState,
                                        AbstractPrediction& head) {
    coverageState.recalculatePrediction(thresholdsSubset, *this, head);
}
