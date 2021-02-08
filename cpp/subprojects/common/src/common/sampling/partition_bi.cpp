#include "common/sampling/partition_bi.hpp"
#include "common/sampling/instance_sampling.hpp"
#include "common/thresholds/coverage_mask.hpp"
#include "common/thresholds/thresholds_subset.hpp"
#include "common/rule_refinement/rule_refinement.hpp"


BiPartition::BiPartition(uint32 numFirst, uint32 numSecond)
    : vector_(DenseVector<uint32>(numFirst + numSecond)), numFirst_(numFirst) {

}

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

std::unique_ptr<IWeightVector> BiPartition::subSample(const IInstanceSubSampling& instanceSubSampling, RNG& rng) const {
    return instanceSubSampling.subSample(*this, rng);
}

void BiPartition::recalculatePrediction(const IThresholdsSubset& thresholdsSubset, const CoverageMask& coverageMask,
                                        Refinement& refinement) const {
    return thresholdsSubset.recalculatePrediction(*this, coverageMask, refinement);
}
