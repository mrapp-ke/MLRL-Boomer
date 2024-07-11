#include "mlrl/common/sampling/instance_sampling_stratified_output_wise.hpp"

#include "mlrl/common/sampling/partition_bi.hpp"
#include "mlrl/common/sampling/partition_single.hpp"
#include "mlrl/common/sampling/stratified_sampling_output_wise.hpp"
#include "mlrl/common/util/validation.hpp"

/**
 * Implements stratified sampling for selecting a subset of the available training examples, such that for each label
 * the proportion of relevant and irrelevant examples is maintained.
 *
 * @tparam LabelMatrix      The type of the label matrix that provides random or row-wise access to the labels of the
 *                          training examples
 * @tparam IndexIterator    The type of the iterator that provides access to the indices of the examples that are
 *                          contained by the training set
 */
template<typename LabelMatrix, typename IndexIterator>
class OutputWiseStratifiedSampling final : public IInstanceSampling {
    private:

        const float32 sampleSize_;

        BitWeightVector weightVector_;

        LabelWiseStratification<LabelMatrix, IndexIterator> stratification_;

    public:

        /**
         * @param labelMatrix   A reference to an object of template type `LabelMatrix` that provides random or
         *                      row-wise access to the labels of the training examples
         * @param indicesBegin  An iterator to the beginning of the indices of the examples that are contained by
         *                      the training set
         * @param indicesEnd    An iterator to the end of the indices of the examples that are contained by the
         *                      training set
         * @param sampleSize    The fraction of examples to be included in the sample (e.g. a value of 0.6
         *                      corresponds to 60 % of the available examples). Must be in (0, 1]
         */
        OutputWiseStratifiedSampling(const LabelMatrix& labelMatrix, IndexIterator indicesBegin,
                                     IndexIterator indicesEnd, float32 sampleSize)
            : sampleSize_(sampleSize),
              weightVector_(labelMatrix.numRows, static_cast<uint32>(indicesEnd - indicesBegin) < labelMatrix.numRows),
              stratification_(labelMatrix, indicesBegin, indicesEnd) {}

        const IWeightVector& sample(RNG& rng) override {
            stratification_.sampleWeights(weightVector_, sampleSize_, rng);
            return weightVector_;
        }
};

/**
 * Allows to create instances of the type `IInstanceSampling` that implement stratified sampling for selecting a subset
 * of the available training examples, such that for each label the proportion of relevant and irrelevant examples is
 * maintained.
 */
class OutputWiseStratifiedInstanceSamplingFactory final : public IInstanceSamplingFactory {
    private:

        const float32 sampleSize_;

    public:

        /**
         * @param sampleSize The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
         *                   60 % of the available examples). Must be in (0, 1]
         */
        OutputWiseStratifiedInstanceSamplingFactory(float32 sampleSize) : sampleSize_(sampleSize) {}

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const uint8>& labelMatrix,
                                                  const SinglePartition& partition,
                                                  IStatistics& statistics) const override {
            return std::make_unique<
              OutputWiseStratifiedSampling<CContiguousView<const uint8>, SinglePartition::const_iterator>>(
              labelMatrix, partition.cbegin(), partition.cend(), sampleSize_);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const uint8>& labelMatrix,
                                                  BiPartition& partition, IStatistics& statistics) const override {
            return std::make_unique<
              OutputWiseStratifiedSampling<CContiguousView<const uint8>, BiPartition::const_iterator>>(
              labelMatrix, partition.first_cbegin(), partition.first_cend(), sampleSize_);
        }

        std::unique_ptr<IInstanceSampling> create(const BinaryCsrView& labelMatrix, const SinglePartition& partition,
                                                  IStatistics& statistics) const override {
            return std::make_unique<OutputWiseStratifiedSampling<BinaryCsrView, SinglePartition::const_iterator>>(
              labelMatrix, partition.cbegin(), partition.cend(), sampleSize_);
        }

        std::unique_ptr<IInstanceSampling> create(const BinaryCsrView& labelMatrix, BiPartition& partition,
                                                  IStatistics& statistics) const override {
            return std::make_unique<OutputWiseStratifiedSampling<BinaryCsrView, BiPartition::const_iterator>>(
              labelMatrix, partition.first_cbegin(), partition.first_cend(), sampleSize_);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const float32>& regressionMatrix,
                                                  const SinglePartition& partition,
                                                  IStatistics& statistics) const override {
            // TODO
            return nullptr;
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const float32>& regressionMatrix,
                                                  BiPartition& partition, IStatistics& statistics) const override {
            // TODO
            return nullptr;
        }

        std::unique_ptr<IInstanceSampling> create(const CsrView<const float32>& regressionMatrix,
                                                  const SinglePartition& partition,
                                                  IStatistics& statistics) const override {
            // TODO
            return nullptr;
        }

        std::unique_ptr<IInstanceSampling> create(const CsrView<const float32>& regressionMatrix,
                                                  BiPartition& partition, IStatistics& statistics) const override {
            // TODO
            return nullptr;
        }
};

OutputWiseStratifiedInstanceSamplingConfig::OutputWiseStratifiedInstanceSamplingConfig() : sampleSize_(0.66f) {}

float32 OutputWiseStratifiedInstanceSamplingConfig::getSampleSize() const {
    return sampleSize_;
}

IOutputWiseStratifiedInstanceSamplingConfig& OutputWiseStratifiedInstanceSamplingConfig::setSampleSize(
  float32 sampleSize) {
    assertGreater<float32>("sampleSize", sampleSize, 0);
    assertLess<float32>("sampleSize", sampleSize, 1);
    sampleSize_ = sampleSize;
    return *this;
}

std::unique_ptr<IInstanceSamplingFactory> OutputWiseStratifiedInstanceSamplingConfig::createInstanceSamplingFactory()
  const {
    return std::make_unique<OutputWiseStratifiedInstanceSamplingFactory>(sampleSize_);
}
