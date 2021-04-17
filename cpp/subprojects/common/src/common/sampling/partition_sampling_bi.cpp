#include "common/sampling/partition_sampling_bi.hpp"
#include "common/sampling/partition_bi.hpp"
#include "index_sampling.hpp"


/**
 * Allows to split the training examples into two mutually exclusive sets that may be used as a training set and a
 * holdout set.
 */
class BiPartitionSampling final : public IPartitionSampling {

    private:

        uint32 numHoldout_;

        uint32 numTraining_;

    public:

        /**
         * @param numExamples       The total number of available training examples
         * @param holdoutSetSize    The fraction of examples to be included in the holdout set (e.g. a value of 0.6
         *                          corresponds to 60 % of the available examples). Must be in (0, 1)
         */
        BiPartitionSampling(uint32 numExamples, float32 holdoutSetSize)
            : numHoldout_((uint32) (holdoutSetSize * numExamples)), numTraining_(numExamples - numHoldout_) {

        }

        std::unique_ptr<IPartition> createPartition(RNG& rng) const override {
            std::unique_ptr<BiPartition> partitionPtr = std::make_unique<BiPartition>(numTraining_, numHoldout_);
            BiPartition::iterator trainingIterator = partitionPtr->first_begin();
            BiPartition::iterator holdoutIterator = partitionPtr->second_begin();

            for (uint32 i = 0; i < numTraining_; i++) {
                trainingIterator[i] = i;
            }

            for (uint32 i = 0; i < numHoldout_; i++) {
                holdoutIterator[i] = numTraining_ + i;
            }

            randomPermutation<BiPartition::iterator, BiPartition::iterator>(trainingIterator, holdoutIterator,
                                                                            numTraining_,
                                                                            partitionPtr->getNumElements(), rng);
            return partitionPtr;
        }

};

BiPartitionSamplingFactory::BiPartitionSamplingFactory(float32 holdoutSetSize)
    : holdoutSetSize_(holdoutSetSize) {

}

std::unique_ptr<IPartitionSampling> BiPartitionSamplingFactory::create(
        const CContiguousLabelMatrix& labelMatrix) const {
    return std::make_unique<BiPartitionSampling>(labelMatrix.getNumRows(), holdoutSetSize_);
}

std::unique_ptr<IPartitionSampling> BiPartitionSamplingFactory::create(const CsrLabelMatrix& labelMatrix) const {
    return std::make_unique<BiPartitionSampling>(labelMatrix.getNumRows(), holdoutSetSize_);
}
