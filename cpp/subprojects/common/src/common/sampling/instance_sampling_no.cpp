#include "common/sampling/instance_sampling_no.hpp"
#include "common/sampling/weight_vector_equal.hpp"
#include "common/sampling/weight_vector_dense.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"


static inline std::unique_ptr<IWeightVector> subSampleInternally(const SinglePartition& partition, RNG& rng) {
    return std::make_unique<EqualWeightVector>(partition.getNumElements());
}

static inline std::unique_ptr<IWeightVector> subSampleInternally(BiPartition& partition, RNG& rng) {
    uint32 numExamples = partition.getNumElements();
    uint32 numTrainingExamples = partition.getNumFirst();
    BiPartition::const_iterator indexIterator = partition.first_cbegin();
    std::unique_ptr<DenseWeightVector<uint8>> weightVectorPtr = std::make_unique<DenseWeightVector<uint8>>(numExamples,
                                                                                                           true);
    typename DenseWeightVector<uint8>::iterator weightIterator = weightVectorPtr->begin();

    for (uint32 i = 0; i < numTrainingExamples; i++) {
        uint32 index = indexIterator[i];
        weightIterator[index] = 1;
    }

    weightVectorPtr->setNumNonZeroWeights(numTrainingExamples);
    return weightVectorPtr;
}

/**
 * An implementation of the class `IInstanceSubSampling` that does not perform any sampling, but assigns equal weights
 * to all examples.
 *
 * @tparam Partition The type of the object that provides access to the indices of the examples that are included in the
 *                   training set
 */
template<class Partition>
class NoInstanceSubSampling final : public IInstanceSubSampling {

    private:

        Partition& partition_;

    public:

        /**
         * @param partition A reference to an object of template type `Partition` that provides access to the indices of
         *                  the examples that are included in the training set
         */
        NoInstanceSubSampling(Partition& partition)
            : partition_(partition) {

        }

        std::unique_ptr<IWeightVector> subSample(RNG& rng) const override {
            return subSampleInternally(partition_, rng);
        }

};

std::unique_ptr<IInstanceSubSampling> NoInstanceSubSamplingFactory::create(
        const CContiguousLabelMatrix& labelMatrix, const SinglePartition& partition) const {
    return std::make_unique<NoInstanceSubSampling<const SinglePartition>>(partition);
}

std::unique_ptr<IInstanceSubSampling> NoInstanceSubSamplingFactory::create(
        const CContiguousLabelMatrix& labelMatrix, BiPartition& partition) const {
    return std::make_unique<NoInstanceSubSampling<BiPartition>>(partition);
}

std::unique_ptr<IInstanceSubSampling> NoInstanceSubSamplingFactory::create(const CsrLabelMatrix& labelMatrix,
                                                                           const SinglePartition& partition) const {
    return std::make_unique<NoInstanceSubSampling<const SinglePartition>>(partition);
}

std::unique_ptr<IInstanceSubSampling> NoInstanceSubSamplingFactory::create(const CsrLabelMatrix& labelMatrix,
                                                                           BiPartition& partition) const {
    return std::make_unique<NoInstanceSubSampling<BiPartition>>(partition);
}
