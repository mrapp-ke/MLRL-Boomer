#include "common/sampling/instance_sampling_no.hpp"
#include "common/sampling/weight_vector_equal.hpp"
#include "common/sampling/weight_vector_dense.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "common/data/arrays.hpp"


static inline void subSampleInternally(const SinglePartition& partition, EqualWeightVector& weightVector, RNG& rng) {
    return;
}

static inline void subSampleInternally(BiPartition& partition, DenseWeightVector<uint8>& weightVector, RNG& rng) {
    uint32 numExamples = partition.getNumElements();
    uint32 numTrainingExamples = partition.getNumFirst();
    BiPartition::const_iterator indexIterator = partition.first_cbegin();
    typename DenseWeightVector<uint8>::iterator weightIterator = weightVector.begin();
    setArrayToZeros(weightIterator, numExamples);

    for (uint32 i = 0; i < numTrainingExamples; i++) {
        uint32 index = indexIterator[i];
        weightIterator[index] = 1;
    }

    weightVector.setNumNonZeroWeights(numTrainingExamples);
}

/**
 * An implementation of the class `IInstanceSubSampling` that does not perform any sampling, but assigns equal weights
 * to all examples.
 *
 * @tparam Partition    The type of the object that provides access to the indices of the examples that are included in
 *                      the training set
 * @tparam WeightVector The type of the weight vector that is used to store the weights
 */
template<class Partition, class WeightVector>
class NoInstanceSubSampling final : public IInstanceSubSampling {

    private:

        Partition& partition_;

        WeightVector weightVector_;

    public:

        /**
         * @param partition A reference to an object of template type `Partition` that provides access to the indices of
         *                  the examples that are included in the training set
         */
        NoInstanceSubSampling(Partition& partition)
            : partition_(partition), weightVector_(WeightVector(partition.getNumElements())) {

        }

        const IWeightVector& subSample(RNG& rng) override {
            subSampleInternally(partition_, weightVector_, rng);
            return weightVector_;
        }

};

std::unique_ptr<IInstanceSubSampling> NoInstanceSubSamplingFactory::create(
        const CContiguousLabelMatrix& labelMatrix, const SinglePartition& partition) const {
    return std::make_unique<NoInstanceSubSampling<const SinglePartition, EqualWeightVector>>(partition);
}

std::unique_ptr<IInstanceSubSampling> NoInstanceSubSamplingFactory::create(
        const CContiguousLabelMatrix& labelMatrix, BiPartition& partition) const {
    return std::make_unique<NoInstanceSubSampling<BiPartition, DenseWeightVector<uint8>>>(partition);
}

std::unique_ptr<IInstanceSubSampling> NoInstanceSubSamplingFactory::create(const CsrLabelMatrix& labelMatrix,
                                                                           const SinglePartition& partition) const {
    return std::make_unique<NoInstanceSubSampling<const SinglePartition, EqualWeightVector>>(partition);
}

std::unique_ptr<IInstanceSubSampling> NoInstanceSubSamplingFactory::create(const CsrLabelMatrix& labelMatrix,
                                                                           BiPartition& partition) const {
    return std::make_unique<NoInstanceSubSampling<BiPartition, DenseWeightVector<uint8>>>(partition);
}
