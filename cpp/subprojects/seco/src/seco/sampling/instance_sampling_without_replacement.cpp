#include "seco/sampling/instance_sampling_without_replacement.hpp"
#include "seco/statistics/statistics.hpp"
#include "seco/data/matrix_dense_weights.hpp"
#include "common/sampling/weight_vector_bit.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "common/sampling/weight_sampling.hpp"


namespace seco {

    static inline uint32 updateExampleIndices(const SinglePartition& partition, const DenseWeightMatrix& weightMatrix,
                                              uint32* exampleIndices) {
        uint32 numTrainingExamples = partition.getNumElements();
        uint32 numLabels = weightMatrix.getNumCols();
        uint32 n = 0;

        for (uint32 i = 0; i < numTrainingExamples; i++) {
            DenseWeightMatrix::const_iterator weightIterator = weightMatrix.row_cbegin(i);

            for (uint32 j = 0; j < numLabels; j++) {
                if (weightIterator[j] > 0) {
                    exampleIndices[n] = i;
                    n++;
                    break;
                }
            }
        }

        return n;
    }

    static inline uint32 updateExampleIndices(BiPartition& partition, const DenseWeightMatrix& weightMatrix,
                                              uint32* exampleIndices) {
        uint32 numTrainingExamples = partition.getNumFirst();
        uint32 numLabels = weightMatrix.getNumCols();
        uint32 n = 0;
        BiPartition::const_iterator indexIterator = partition.first_cbegin();

        for (uint32 i = 0; i < numTrainingExamples; i++) {
            uint32 index = indexIterator[i];
            DenseWeightMatrix::const_iterator weightIterator = weightMatrix.row_cbegin(index);

            for (uint32 j = 0; j < numLabels; j++) {
                if (weightIterator[j] > 0) {
                    exampleIndices[n] = index;
                    n++;
                    break;
                }
            }
        }

        return n;
    }

    /**
     * Allows to select a subset of the available training examples that have at least one label with non-zero weight
     * without replacement.
     *
     * @tparam Partition    The type of the object that provides access to the indices of the examples that are included
     *                      in the training set
     * @tparam WeightMatrix The type of the matrix that provides access to the weights of individual examples and labels
     */
    template<typename Partition, typename WeightMatrix>
    class InstanceSamplingWithoutReplacement final : public IInstanceSampling {

        private:

            Partition& partition_;

            const WeightMatrix& weightMatrix_;

            float32 sampleSize_;

            BitWeightVector weightVector_;

            uint32* exampleIndices_;

        public:

            /**
             * @param partition     A reference to an object of template type `Partition` that provides access to the
             *                      indices of the examples that are included in the training set
             * @param weightMatrix  A reference to an object of template type `WeightMatrix` that provides access to the
             *                      weights of individual examples and labels
             * @param sampleSize    The fraction of examples to be included in the sample (e.g. a value of 0.6
             *                      corresponds to 60 % of the available examples). Must be in (0, 1)
             */
            InstanceSamplingWithoutReplacement(Partition& partition, const WeightMatrix& weightMatrix,
                                               float32 sampleSize)
                : partition_(partition), weightMatrix_(weightMatrix), sampleSize_(sampleSize),
                  weightVector_(BitWeightVector(partition.getNumElements())),
                  exampleIndices_(new uint32[partition.getNumElements()]) {

            }

            ~InstanceSamplingWithoutReplacement() {
                delete[] exampleIndices_;
            }

            const IWeightVector& sample(RNG& rng) override {
                uint32 numExampleIndices = updateExampleIndices(partition_, weightMatrix_, exampleIndices_);
                uint32 numSamples = (uint32) (sampleSize_ * numExampleIndices);
                sampleWeightsWithoutReplacement<const uint32*>(weightVector_, exampleIndices_, numExampleIndices,
                                                               numSamples, rng);
                return weightVector_;
            }

    };

    template<typename Partition>
    static inline std::unique_ptr<IInstanceSampling> createSampling(Partition& partition, IStatistics& statistics,
                                                                    float32 sampleSize) {
        std::unique_ptr<IInstanceSampling> instanceSamplingPtr;
        ICoverageStatistics::DenseWeightMatrixVisitor denseWeightMatrixVisitor =
            [&](std::unique_ptr<DenseWeightMatrix>& weightMatrixPtr) mutable {
                instanceSamplingPtr =
                    std::make_unique<InstanceSamplingWithoutReplacement<Partition, DenseWeightMatrix>>(partition,
                                                                                                       *weightMatrixPtr,
                                                                                                       sampleSize);
        };
        ICoverageStatistics& coverageStatistics = dynamic_cast<ICoverageStatistics&>(statistics);
        coverageStatistics.visitWeightMatrix(denseWeightMatrixVisitor);
        return instanceSamplingPtr;
    }

    InstanceSamplingWithoutReplacementFactory::InstanceSamplingWithoutReplacementFactory(float32 sampleSize)
        : sampleSize_(sampleSize) {

    }

    std::unique_ptr<IInstanceSampling> InstanceSamplingWithoutReplacementFactory::create(
            const CContiguousLabelMatrix& labelMatrix, const SinglePartition& partition,
            IStatistics& statistics) const {
        return createSampling<const SinglePartition>(partition, statistics, sampleSize_);
    }

    std::unique_ptr<IInstanceSampling> InstanceSamplingWithoutReplacementFactory::create(
            const CContiguousLabelMatrix& labelMatrix, BiPartition& partition, IStatistics& statistics) const {
        return createSampling<BiPartition>(partition, statistics, sampleSize_);
    }

    std::unique_ptr<IInstanceSampling> InstanceSamplingWithoutReplacementFactory::create(
            const CsrLabelMatrix& labelMatrix, const SinglePartition& partition, IStatistics& statistics) const {
        return createSampling<const SinglePartition>(partition, statistics, sampleSize_);
    }

    std::unique_ptr<IInstanceSampling> InstanceSamplingWithoutReplacementFactory::create(
            const CsrLabelMatrix& labelMatrix, BiPartition& partition, IStatistics& statistics) const {
        return createSampling<BiPartition>(partition, statistics, sampleSize_);
    }

}
