#include "mlrl/common/sampling/instance_sampling_no.hpp"

#include "mlrl/common/sampling/partition_bi.hpp"
#include "mlrl/common/sampling/partition_single.hpp"
#include "mlrl/common/sampling/weight_vector_bit.hpp"
#include "mlrl/common/sampling/weight_vector_equal.hpp"

static inline void sampleInternally(const SinglePartition& partition, EqualWeightVector& weightVector) {
    return;
}

static inline void sampleInternally(BiPartition& partition, BitWeightVector& weightVector) {
    uint32 numTrainingExamples = partition.getNumFirst();
    BiPartition::const_iterator indexIterator = partition.first_cbegin();
    weightVector.clear();

    for (uint32 i = 0; i < numTrainingExamples; i++) {
        uint32 index = indexIterator[i];
        weightVector.set(index, true);
    }

    weightVector.setNumNonZeroWeights(numTrainingExamples);
}

/**
 * An implementation of the class `IInstanceSampling` that does not perform any sampling, but assigns equal weights to
 * all examples.
 *
 * @tparam Partition    The type of the object that provides access to the indices of the examples that are included in
 *                      the training set
 * @tparam WeightVector The type of the weight vector that is used to store the weights
 */
template<typename Partition, typename WeightVector>
class NoInstanceSampling final : public IInstanceSampling {
    private:

        Partition& partition_;

        WeightVector weightVector_;

    public:

        /**
         * @param partition A reference to an object of template type `Partition` that provides access to the indices of
         *                  the examples that are included in the training set
         */
        NoInstanceSampling(Partition& partition) : partition_(partition), weightVector_(partition.getNumElements()) {}

        const IWeightVector& sample() override {
            sampleInternally(partition_, weightVector_);
            return weightVector_;
        }
};

template<typename Partition, typename WeightVector>
static inline std::unique_ptr<IInstanceSampling> createNoInstanceSampling(Partition& partition) {
    return std::make_unique<NoInstanceSampling<Partition, WeightVector>>(partition);
}

/**
 * Allows to create instances of the type `IInstanceSampling` that do not perform any sampling, but assign equal weights
 * to all examples.
 */
class NoInstanceSamplingFactory final : public IClassificationInstanceSamplingFactory,
                                        public IRegressionInstanceSamplingFactory {
    public:

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const uint8>& labelMatrix,
                                                  const SinglePartition& partition, IStatistics& statistics,
                                                  const EqualWeightVector& exampleWeights) const override {
            return createNoInstanceSampling<const SinglePartition, EqualWeightVector>(partition);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const uint8>& labelMatrix,
                                                  const SinglePartition& partition, IStatistics& statistics,
                                                  const DenseWeightVector<float32>& exampleWeights) const override {
            return createNoInstanceSampling<const SinglePartition, EqualWeightVector>(partition);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const uint8>& labelMatrix,
                                                  BiPartition& partition, IStatistics& statistics,
                                                  const EqualWeightVector& exampleWeights) const override {
            return createNoInstanceSampling<BiPartition, BitWeightVector>(partition);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const uint8>& labelMatrix,
                                                  BiPartition& partition, IStatistics& statistics,
                                                  const DenseWeightVector<float32>& exampleWeights) const override {
            return createNoInstanceSampling<BiPartition, BitWeightVector>(partition);
        }

        std::unique_ptr<IInstanceSampling> create(const BinaryCsrView& labelMatrix, const SinglePartition& partition,
                                                  IStatistics& statistics,
                                                  const EqualWeightVector& exampleWeights) const override {
            return createNoInstanceSampling<const SinglePartition, EqualWeightVector>(partition);
        }

        std::unique_ptr<IInstanceSampling> create(const BinaryCsrView& labelMatrix, const SinglePartition& partition,
                                                  IStatistics& statistics,
                                                  const DenseWeightVector<float32>& exampleWeights) const override {
            return createNoInstanceSampling<const SinglePartition, EqualWeightVector>(partition);
        }

        std::unique_ptr<IInstanceSampling> create(const BinaryCsrView& labelMatrix, BiPartition& partition,
                                                  IStatistics& statistics,
                                                  const EqualWeightVector& exampleWeights) const override {
            return createNoInstanceSampling<BiPartition, BitWeightVector>(partition);
        }

        std::unique_ptr<IInstanceSampling> create(const BinaryCsrView& labelMatrix, BiPartition& partition,
                                                  IStatistics& statistics,
                                                  const DenseWeightVector<float32>& exampleWeights) const override {
            return createNoInstanceSampling<BiPartition, BitWeightVector>(partition);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const float32>& regressionMatrix,
                                                  const SinglePartition& partition, IStatistics& statistics,
                                                  const EqualWeightVector& exampleWeights) const override {
            return createNoInstanceSampling<const SinglePartition, EqualWeightVector>(partition);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const float32>& regressionMatrix,
                                                  const SinglePartition& partition, IStatistics& statistics,
                                                  const DenseWeightVector<float32>& exampleWeights) const override {
            return createNoInstanceSampling<const SinglePartition, EqualWeightVector>(partition);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const float32>& regressionMatrix,
                                                  BiPartition& partition, IStatistics& statistics,
                                                  const EqualWeightVector& exampleWeights) const override {
            return createNoInstanceSampling<BiPartition, BitWeightVector>(partition);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const float32>& regressionMatrix,
                                                  BiPartition& partition, IStatistics& statistics,
                                                  const DenseWeightVector<float32>& exampleWeights) const override {
            return createNoInstanceSampling<BiPartition, BitWeightVector>(partition);
        }

        std::unique_ptr<IInstanceSampling> create(const CsrView<const float32>& regressionMatrix,
                                                  const SinglePartition& partition, IStatistics& statistics,
                                                  const EqualWeightVector& exampleWeights) const override {
            return createNoInstanceSampling<const SinglePartition, EqualWeightVector>(partition);
        }

        std::unique_ptr<IInstanceSampling> create(const CsrView<const float32>& regressionMatrix,
                                                  const SinglePartition& partition, IStatistics& statistics,
                                                  const DenseWeightVector<float32>& exampleWeights) const override {
            return createNoInstanceSampling<const SinglePartition, EqualWeightVector>(partition);
        }

        std::unique_ptr<IInstanceSampling> create(const CsrView<const float32>& regressionMatrix,
                                                  BiPartition& partition, IStatistics& statistics,
                                                  const EqualWeightVector& exampleWeights) const override {
            return createNoInstanceSampling<BiPartition, BitWeightVector>(partition);
        }

        std::unique_ptr<IInstanceSampling> create(const CsrView<const float32>& regressionMatrix,
                                                  BiPartition& partition, IStatistics& statistics,
                                                  const DenseWeightVector<float32>& exampleWeights) const override {
            return createNoInstanceSampling<BiPartition, BitWeightVector>(partition);
        }
};

std::unique_ptr<IClassificationInstanceSamplingFactory>
  NoInstanceSamplingConfig::createClassificationInstanceSamplingFactory() const {
    return std::make_unique<NoInstanceSamplingFactory>();
}

std::unique_ptr<IRegressionInstanceSamplingFactory> NoInstanceSamplingConfig::createRegressionInstanceSamplingFactory()
  const {
    return std::make_unique<NoInstanceSamplingFactory>();
}
