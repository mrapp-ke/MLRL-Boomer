#include "common/sampling/instance_sampling_stratified_label_wise.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"


/**
 * Implements bootstrap aggregating (bagging) for selecting a subset of the available training examples with
 * replacement.
 */
class LabelWiseStratifiedSampling final : public IInstanceSubSampling {

    public:

        std::unique_ptr<IWeightVector> subSample(const SinglePartition& partition, RNG& rng) const override {
            // TODO Implement
            return nullptr;
        }

        std::unique_ptr<IWeightVector> subSample(const BiPartition& partition, RNG& rng) const override {
            // TODO Implement
            return nullptr;
        }

};

std::unique_ptr<IInstanceSubSampling> LabelWiseStratifiedSamplingFactory::create(
        const CContiguousLabelMatrix& labelMatrix) const {
    return std::make_unique<LabelWiseStratifiedSampling>();
}

std::unique_ptr<IInstanceSubSampling> LabelWiseStratifiedSamplingFactory::create(
        const CsrLabelMatrix& labelMatrix) const {
    return std::make_unique<LabelWiseStratifiedSampling>();
}
