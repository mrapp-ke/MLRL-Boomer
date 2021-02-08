#include "common/sampling/partition_single.hpp"
#include "common/sampling/instance_sampling.hpp"
#include "common/thresholds/coverage_mask.hpp"
#include "common/thresholds/thresholds_subset.hpp"
#include "common/rule_refinement/rule_refinement.hpp"


SinglePartition::SinglePartition(uint32 numElements)
    : numElements_(numElements) {

}

SinglePartition::const_iterator SinglePartition::cbegin() const {
    return IndexIterator(0);
}

SinglePartition::const_iterator SinglePartition::cend() const {
    return IndexIterator(numElements_);
}

uint32 SinglePartition::getNumElements() const {
    return numElements_;
}

std::unique_ptr<IWeightVector> SinglePartition::subSample(const IInstanceSubSampling& instanceSubSampling,
                                                          RNG& rng) const {
    return instanceSubSampling.subSample(*this, rng);
}

void SinglePartition::recalculatePrediction(const IThresholdsSubset& thresholdsSubset, const CoverageMask& coverageMask,
                                            Refinement& refinement) const {
    thresholdsSubset.recalculatePrediction(*this, coverageMask, refinement);
}
