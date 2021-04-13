#include "common/sampling/instance_sampling_stratified_label_wise.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"


/**
 * Implements iterative stratified sampling for selecting a subset of the available training examples, such that for
 * each label the proportion of relevant and irrelevant examples is maintained.
 */
class LabelWiseStratifiedSampling final : public IInstanceSubSampling {

    private:

        float32 sampleSize_;

    public:

        /**
         * @param sampleSize The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
         *                   60 % of the available examples). Must be in (0, 1]
         */
        LabelWiseStratifiedSampling(float32 sampleSize)
            : sampleSize_(sampleSize) {

        }

        std::unique_ptr<IWeightVector> subSample(const SinglePartition& partition, RNG& rng) const override {
            // TODO Implement
            return nullptr;
        }

        std::unique_ptr<IWeightVector> subSample(const BiPartition& partition, RNG& rng) const override {
            // TODO Implement
            return nullptr;
        }

};

LabelWiseStratifiedSamplingFactory::LabelWiseStratifiedSamplingFactory(float32 sampleSize)
    : sampleSize_(sampleSize) {

}

std::unique_ptr<IInstanceSubSampling> LabelWiseStratifiedSamplingFactory::create(
        const CContiguousLabelMatrix& labelMatrix) const {
    return std::make_unique<LabelWiseStratifiedSampling>(sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> LabelWiseStratifiedSamplingFactory::create(
        const CsrLabelMatrix& labelMatrix) const {
    return std::make_unique<LabelWiseStratifiedSampling>(sampleSize_);
}
