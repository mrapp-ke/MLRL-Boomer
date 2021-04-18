#include "common/sampling/partition_sampling_bi.hpp"
#include "common/sampling/partition_bi.hpp"
#include "index_sampling.hpp"


/**
 * Allows to split the training examples into two mutually exclusive sets that may be used as a training set and a
 * holdout set.
 */
class BiPartitionSampling final : public IPartitionSampling {

    private:

        BiPartition partition_;

    public:

        /**
         * @param numExamples       The total number of available training examples
         * @param holdoutSetSize    The fraction of examples to be included in the holdout set (e.g. a value of 0.6
         *                          corresponds to 60 % of the available examples). Must be in (0, 1)
         */
        BiPartitionSampling(uint32 numExamples, float32 holdoutSetSize)
            : partition_(BiPartition(numExamples - ((uint32) holdoutSetSize * numExamples),
                                     (uint32) (holdoutSetSize * numExamples))) {

        }

        IPartition& partition(RNG& rng) override {
            uint32 numTraining = partition_.getNumFirst();
            uint32 numHoldout = partition_.getNumSecond();
            BiPartition::iterator trainingIterator = partition_.first_begin();
            BiPartition::iterator holdoutIterator = partition_.second_begin();

            for (uint32 i = 0; i < numTraining; i++) {
                trainingIterator[i] = i;
            }

            for (uint32 i = 0; i < numHoldout; i++) {
                holdoutIterator[i] = numTraining + i;
            }

            uint32 numTotal = partition_.getNumElements();
            randomPermutation<BiPartition::iterator, BiPartition::iterator>(trainingIterator, holdoutIterator,
                                                                            numTraining, numTotal, numTraining, rng);
            return partition_;
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
