#include "common/sampling/instance_sampling_bagging.hpp"
#include "common/sampling/weight_vector_dense.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"


/**
 * Implements bootstrap aggregating (bagging) for selecting a subset of the available training examples with
 * replacement.
 */
class Bagging final : public IInstanceSubSampling {

    private:

        float32 sampleSize_;

    public:

        /**
         * @param sampleSize The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
         *                   60 % of the available examples). Must be in (0, 1]
         */
        Bagging(float32 sampleSize)
            : sampleSize_(sampleSize) {

        }

        std::unique_ptr<IWeightVector> subSample(const SinglePartition& partition, RNG& rng) const override {
            uint32 numExamples = partition.getNumElements();
            uint32 numSamples = (uint32) (sampleSize_ * numExamples);
            std::unique_ptr<DenseWeightVector<uint32>> weightVectorPtr = std::make_unique<DenseWeightVector<uint32>>(
                numExamples);
            typename DenseWeightVector<uint32>::iterator weightIterator = weightVectorPtr->begin();
            uint32 numNonZeroWeights = 0;

            for (uint32 i = 0; i < numSamples; i++) {
                // Randomly select the index of an example...
                uint32 randomIndex = rng.random(0, numExamples);

                // Update weight at the selected index...
                uint32 previousWeight = weightIterator[randomIndex];
                weightIterator[randomIndex] = previousWeight + 1;

                if (previousWeight == 0) {
                    numNonZeroWeights++;
                }
            }

            weightVectorPtr->setNumNonZeroWeights(numNonZeroWeights);
            return weightVectorPtr;
        }

        std::unique_ptr<IWeightVector> subSample(const BiPartition& partition, RNG& rng) const override {
            uint32 numExamples = partition.getNumElements();
            uint32 numTrainingExamples = partition.getNumFirst();
            uint32 numSamples = (uint32) (sampleSize_ * numTrainingExamples);
            BiPartition::const_iterator indexIterator = partition.first_cbegin();
            std::unique_ptr<DenseWeightVector<uint32>> weightVectorPtr = std::make_unique<DenseWeightVector<uint32>>(
                numExamples);
            typename DenseWeightVector<uint32>::iterator weightIterator = weightVectorPtr->begin();
            uint32 numNonZeroWeights = 0;

            for (uint32 i = 0; i < numSamples; i++) {
                // Randomly select the index of an example...
                uint32 randomIndex = rng.random(0, numTrainingExamples);
                uint32 sampledIndex = indexIterator[randomIndex];

                // Update weight at the selected index...
                uint32 previousWeight = weightIterator[sampledIndex];
                weightIterator[sampledIndex] = previousWeight + 1;

                if (previousWeight == 0) {
                    numNonZeroWeights++;
                }
            }

            weightVectorPtr->setNumNonZeroWeights(numNonZeroWeights);
            return weightVectorPtr;
        }

};

BaggingFactory::BaggingFactory(float32 sampleSize)
    : sampleSize_(sampleSize) {

}

std::unique_ptr<IInstanceSubSampling> BaggingFactory::create(const CContiguousLabelMatrix& labelMatrix) const {
    return std::make_unique<Bagging>(sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> BaggingFactory::create(const CsrLabelMatrix& labelMatrix) const {
    return std::make_unique<Bagging>(sampleSize_);
}
