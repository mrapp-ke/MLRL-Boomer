#include "common/sampling/instance_sampling_stratified_example_wise.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"


/**
 * Implements stratified sampling, where distinct label vectors are treated as individual classes.
 */
class ExampleWiseStratifiedSampling final : public IInstanceSubSampling {

    private:

        float32 sampleSize_;

    public:

        /**
         * @param sampleSize The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
         *                   60 % of the available examples). Must be in (0, 1]
         */
        ExampleWiseStratifiedSampling(float32 sampleSize)
            : sampleSize_(sampleSize) {

        }

        std::unique_ptr<IWeightVector> subSample(RNG& rng) override {
            // TODO Implement
            return nullptr;
        }

};

ExampleWiseStratifiedSamplingFactory::ExampleWiseStratifiedSamplingFactory(float32 sampleSize)
    : sampleSize_(sampleSize) {

}

std::unique_ptr<IInstanceSubSampling> ExampleWiseStratifiedSamplingFactory::create(
        const CContiguousLabelMatrix& labelMatrix, const SinglePartition& partition) const {
    return std::make_unique<ExampleWiseStratifiedSampling>(sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> ExampleWiseStratifiedSamplingFactory::create(
        const CContiguousLabelMatrix& labelMatrix, BiPartition& partition) const {
    return std::make_unique<ExampleWiseStratifiedSampling>(sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> ExampleWiseStratifiedSamplingFactory::create(
        const CsrLabelMatrix& labelMatrix, const SinglePartition& partition) const {
    return std::make_unique<ExampleWiseStratifiedSampling>(sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> ExampleWiseStratifiedSamplingFactory::create(
        const CsrLabelMatrix& labelMatrix, BiPartition& partition) const {
    return std::make_unique<ExampleWiseStratifiedSampling>(sampleSize_);
}
