#include "common/sampling/instance_sampling_no.hpp"
#include "common/sampling/weight_vector_equal.hpp"
#include "common/sampling/weight_vector_dense.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"


/**
 * An implementation of the class `IInstanceSubSampling` that does not perform any sampling, but assigns equal weights
 * to all examples.
 */
class NoInstanceSubSampling final : public IInstanceSubSampling {

    public:

        std::unique_ptr<IWeightVector> subSample(const SinglePartition& partition, RNG& rng) const override {
            return std::make_unique<EqualWeightVector>(partition.getNumElements());
        }

        std::unique_ptr<IWeightVector> subSample(const BiPartition& partition, RNG& rng) const override {
            uint32 numExamples = partition.getNumElements();
            uint32 numTrainingExamples = partition.getNumFirst();
            BiPartition::const_iterator indexIterator = partition.first_cbegin();
            std::unique_ptr<DenseWeightVector<uint32>> weightVectorPtr = std::make_unique<DenseWeightVector<uint32>>(
                numExamples);
            typename DenseWeightVector<uint32>::iterator weightIterator = weightVectorPtr->begin();

            for (uint32 i = 0; i < numTrainingExamples; i++) {
                uint32 index = indexIterator[i];
                weightIterator[index] = 1;
            }

            weightVectorPtr->setNumNonZeroWeights(numTrainingExamples);
            return weightVectorPtr;
        }

};

std::unique_ptr<IInstanceSubSampling> NoInstanceSubSamplingFactory::create(
        const CContiguousLabelMatrix& labelMatrix) const {
    return std::make_unique<NoInstanceSubSampling>();
}

std::unique_ptr<IInstanceSubSampling> NoInstanceSubSamplingFactory::create(const CsrLabelMatrix& labelMatrix) const {
    return std::make_unique<NoInstanceSubSampling>();
}
