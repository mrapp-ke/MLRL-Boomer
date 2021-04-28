#include "common/sampling/partition_sampling_bi_stratified_example_wise.hpp"
#include "common/sampling/partition_bi.hpp"


/**
 * Allows to use stratified sampling, where distinct label vectors are treated as individual classes, to split the
 * training examples into two mutually exclusive sets that may be used as a training set and a holdout set.
 */
class ExampleWiseStratifiedBiPartitionSampling final : public IPartitionSampling {

    private:

        BiPartition partition_;

    public:

        /**
         * @param numExamples       The total number of available training examples
         * @param holdoutSetSize    The fraction of examples to be included in the holdout set (e.g. a value of 0.6
         *                          corresponds to 60 % of the available examples). Must be in (0, 1)
         */
        ExampleWiseStratifiedBiPartitionSampling(uint32 numExamples, float32 holdoutSetSize)
            : partition_(BiPartition(numExamples - ((uint32) holdoutSetSize * numExamples),
                                     (uint32) (holdoutSetSize * numExamples))) {

        }

        IPartition& partition(RNG& rng) override {
            // TODO Implement
            return partition_;
        }

};

ExampleWiseStratifiedBiPartitionSamplingFactory::ExampleWiseStratifiedBiPartitionSamplingFactory(float32 holdoutSetSize)
    : holdoutSetSize_(holdoutSetSize) {

}

std::unique_ptr<IPartitionSampling> ExampleWiseStratifiedBiPartitionSamplingFactory::create(
        const CContiguousLabelMatrix& labelMatrix) const {
    return std::make_unique<ExampleWiseStratifiedBiPartitionSampling>(labelMatrix.getNumRows(), holdoutSetSize_);
}

std::unique_ptr<IPartitionSampling> ExampleWiseStratifiedBiPartitionSamplingFactory::create(
        const CsrLabelMatrix& labelMatrix) const {
    return std::make_unique<ExampleWiseStratifiedBiPartitionSampling>(labelMatrix.getNumRows(), holdoutSetSize_);
}
