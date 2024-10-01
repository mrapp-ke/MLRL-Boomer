#include "mlrl/common/sampling/instance_sampling_stratified_example_wise.hpp"

#include "mlrl/common/sampling/partition_bi.hpp"
#include "mlrl/common/sampling/partition_single.hpp"
#include "mlrl/common/sampling/stratified_sampling_example_wise.hpp"
#include "mlrl/common/util/validation.hpp"

/**
 * Implements stratified sampling, where distinct label vectors are treated as individual classes.
 *
 * @tparam LabelMatrix      The type of the label matrix that provides random or row-wise access to the labels of the
 *                          training examples
 * @tparam IndexIterator    The type of the iterator that provides access to the indices of the examples that are
 *                          contained by the training set
 */
template<typename LabelMatrix, typename IndexIterator>
class ExampleWiseStratifiedSampling final : public IInstanceSampling {
    private:

        const float32 sampleSize_;

        BitWeightVector weightVector_;

        const ExampleWiseStratification<LabelMatrix, IndexIterator> stratification_;

    public:

        /**
         * @param rngPtr        An unique pointer to an object of type `RNG` that should be used for generating random
         *                      numbers
         * @param labelMatrix   A reference to an object of template type `LabelMatrix` that provides random or
         *                      row-wise access to the labels of the training examples
         * @param indicesBegin  An iterator to the beginning of the indices of the examples that are contained by
         *                      the training set
         * @param indicesEnd    An iterator to the end of the indices of the examples that are contained by the
         *                      training set
         * @param sampleSize    The fraction of examples to be included in the sample (e.g. a value of 0.6
         *                      corresponds to 60 % of the available examples). Must be in (0, 1]
         */
        ExampleWiseStratifiedSampling(std::unique_ptr<RNG> rngPtr, const LabelMatrix& labelMatrix,
                                      IndexIterator indicesBegin, IndexIterator indicesEnd, float32 sampleSize)
            : sampleSize_(sampleSize),
              weightVector_(labelMatrix.numRows, static_cast<uint32>(indicesEnd - indicesBegin) < labelMatrix.numRows),
              stratification_(std::move(rngPtr), labelMatrix, indicesBegin, indicesEnd) {}

        const IWeightVector& sample() override {
            stratification_.sampleWeights(weightVector_, sampleSize_, 1, 0);
            return weightVector_;
        }
};

/**
 * Allows to create instances of the type `IInstanceSampling` that implement stratified sampling, where distinct label
 * vectors are treated as individual classes.
 */
class ExampleWiseStratifiedInstanceSamplingFactory final : public IClassificationInstanceSamplingFactory {
    private:

        std::unique_ptr<RNGFactory> rngFactoryPtr_;

        const float32 sampleSize_;

    public:

        /**
         * @param rngFactoryPtr An unique pointer to an object of type `RNGFactory` that allows to create random number
         *                      generators
         * @param sampleSize    The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds
         *                      to 60 % of the available examples). Must be in (0, 1]
         */
        ExampleWiseStratifiedInstanceSamplingFactory(std::unique_ptr<RNGFactory> rngFactoryPtr, float32 sampleSize)
            : rngFactoryPtr_(std::move(rngFactoryPtr)), sampleSize_(sampleSize) {}

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const uint8>& labelMatrix,
                                                  const SinglePartition& partition,
                                                  IStatistics& statistics) const override {
            return std::make_unique<
              ExampleWiseStratifiedSampling<CContiguousView<const uint8>, SinglePartition::const_iterator>>(
              rngFactoryPtr_->create(), labelMatrix, partition.cbegin(), partition.cend(), sampleSize_);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const uint8>& labelMatrix,
                                                  BiPartition& partition, IStatistics& statistics) const override {
            return std::make_unique<
              ExampleWiseStratifiedSampling<CContiguousView<const uint8>, BiPartition::const_iterator>>(
              rngFactoryPtr_->create(), labelMatrix, partition.first_cbegin(), partition.first_cend(), sampleSize_);
        }

        std::unique_ptr<IInstanceSampling> create(const BinaryCsrView& labelMatrix, const SinglePartition& partition,
                                                  IStatistics& statistics) const override {
            return std::make_unique<ExampleWiseStratifiedSampling<BinaryCsrView, SinglePartition::const_iterator>>(
              rngFactoryPtr_->create(), labelMatrix, partition.cbegin(), partition.cend(), sampleSize_);
        }

        std::unique_ptr<IInstanceSampling> create(const BinaryCsrView& labelMatrix, BiPartition& partition,
                                                  IStatistics& statistics) const override {
            return std::make_unique<ExampleWiseStratifiedSampling<BinaryCsrView, BiPartition::const_iterator>>(
              rngFactoryPtr_->create(), labelMatrix, partition.first_cbegin(), partition.first_cend(), sampleSize_);
        }
};

ExampleWiseStratifiedInstanceSamplingConfig::ExampleWiseStratifiedInstanceSamplingConfig(
  ReadableProperty<RNGConfig> rngConfig)
    : rngConfig_(rngConfig), sampleSize_(0.66f) {}

float32 ExampleWiseStratifiedInstanceSamplingConfig::getSampleSize() const {
    return sampleSize_;
}

IExampleWiseStratifiedInstanceSamplingConfig& ExampleWiseStratifiedInstanceSamplingConfig::setSampleSize(
  float32 sampleSize) {
    util::assertGreater<float32>("sampleSize", sampleSize, 0);
    util::assertLess<float32>("sampleSize", sampleSize, 1);
    sampleSize_ = sampleSize;
    return *this;
}

std::unique_ptr<IClassificationInstanceSamplingFactory>
  ExampleWiseStratifiedInstanceSamplingConfig::createClassificationInstanceSamplingFactory() const {
    return std::make_unique<ExampleWiseStratifiedInstanceSamplingFactory>(rngConfig_.get().createRNGFactory(),
                                                                          sampleSize_);
}
