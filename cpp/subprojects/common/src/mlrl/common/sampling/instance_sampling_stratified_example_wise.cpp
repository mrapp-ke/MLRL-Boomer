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
 * @tparam ExampleWeights   The type of the object that provides access to the weights of individual training examples
 * @tparam WeightVector     The type of the vector, the weights of the sampled examples should be written to
 */
template<typename LabelMatrix, typename IndexIterator, typename ExampleWeights, typename WeightVector>
class ExampleWiseStratifiedSampling final : public IInstanceSampling {
    private:

        const ExampleWeights exampleWeights_;

        const float32 sampleSize_;

        const uint32 minSamples_;

        const uint32 maxSamples_;

        WeightVector weightVector_;

        const ExampleWiseStratification<LabelMatrix, IndexIterator> stratification_;

    public:

        /**
         * @param rngPtr          An unique pointer to an object of type `RNG` that should be used for generating random
         *                        numbers
         * @param labelMatrix     A reference to an object of template type `LabelMatrix` that provides random or
         *                        row-wise access to the labels of the training examples
         * @param indicesBegin    An iterator to the beginning of the indices of the examples that are contained by
         *                        the training set
         * @param indicesEnd      An iterator to the end of the indices of the examples that are contained by the
         *                        training set
         * @param exampleWeights  An object of template type `ExampleWeights` that provides access to the weights of
         *                        individual training examples
         * @param sampleSize      The fraction of examples to be included in the sample (e.g. a value of 0.6
         *                        corresponds to 60 % of the available examples). Must be in (0, 1]
         * @param minSamples      The minimum number of examples to be included in the sample. Must be at least 1
         * @param maxSamples      The maximum number of examples to be included in the sample. Must be at least
         *                        `minSamples` or 0, if the number of examples should not be restricted
         */
        ExampleWiseStratifiedSampling(std::unique_ptr<RNG> rngPtr, const LabelMatrix& labelMatrix,
                                      IndexIterator indicesBegin, IndexIterator indicesEnd,
                                      const ExampleWeights& exampleWeights, float32 sampleSize, uint32 minSamples,
                                      uint32 maxSamples)
            : exampleWeights_(exampleWeights), sampleSize_(sampleSize), minSamples_(minSamples),
              maxSamples_(maxSamples),
              weightVector_(labelMatrix.numRows, static_cast<uint32>(indicesEnd - indicesBegin) < labelMatrix.numRows),
              stratification_(std::move(rngPtr), labelMatrix, indicesBegin, indicesEnd) {}

        const IWeightVector& sample() override {
            stratification_.sampleWeights(weightVector_, exampleWeights_.cbegin(), sampleSize_, minSamples_,
                                          maxSamples_);
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

        const uint32 minSamples_;

        const uint32 maxSamples_;

    public:

        /**
         * @param rngFactoryPtr An unique pointer to an object of type `RNGFactory` that allows to create random number
         *                      generators
         * @param sampleSize    The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds
         *                      to 60 % of the available examples). Must be in (0, 1]
         * @param minSamples    The minimum number of examples to be included in the sample. Must be at least 1
         * @param maxSamples    The maximum number of examples to be included in the sample. Must be at least
         *                      `minSamples` or 0, if the number of examples should not be restricted
         */
        ExampleWiseStratifiedInstanceSamplingFactory(std::unique_ptr<RNGFactory> rngFactoryPtr, float32 sampleSize,
                                                     uint32 minSamples, uint32 maxSamples)
            : rngFactoryPtr_(std::move(rngFactoryPtr)), sampleSize_(sampleSize), minSamples_(minSamples),
              maxSamples_(maxSamples) {}

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const uint8>& labelMatrix,
                                                  const SinglePartition& partition, IStatistics& statistics,
                                                  const EqualWeightVector& exampleWeights) const override {
            return std::make_unique<ExampleWiseStratifiedSampling<
              CContiguousView<const uint8>, SinglePartition::const_iterator, EqualWeightVector, BitWeightVector>>(
              rngFactoryPtr_->create(), labelMatrix, partition.cbegin(), partition.cend(), exampleWeights, sampleSize_,
              minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const uint8>& labelMatrix,
                                                  const SinglePartition& partition, IStatistics& statistics,
                                                  const DenseWeightVector<float32>& exampleWeights) const override {
            return std::make_unique<
              ExampleWiseStratifiedSampling<CContiguousView<const uint8>, SinglePartition::const_iterator,
                                            DenseWeightVector<float32>, DenseWeightVector<float32>>>(
              rngFactoryPtr_->create(), labelMatrix, partition.cbegin(), partition.cend(), exampleWeights, sampleSize_,
              minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const uint8>& labelMatrix,
                                                  BiPartition& partition, IStatistics& statistics,
                                                  const EqualWeightVector& exampleWeights) const override {
            return std::make_unique<ExampleWiseStratifiedSampling<
              CContiguousView<const uint8>, BiPartition::const_iterator, EqualWeightVector, BitWeightVector>>(
              rngFactoryPtr_->create(), labelMatrix, partition.first_cbegin(), partition.first_cend(), exampleWeights,
              sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousView<const uint8>& labelMatrix,
                                                  BiPartition& partition, IStatistics& statistics,
                                                  const DenseWeightVector<float32>& exampleWeights) const override {
            return std::make_unique<
              ExampleWiseStratifiedSampling<CContiguousView<const uint8>, BiPartition::const_iterator,
                                            DenseWeightVector<float32>, DenseWeightVector<float32>>>(
              rngFactoryPtr_->create(), labelMatrix, partition.first_cbegin(), partition.first_cend(), exampleWeights,
              sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const BinaryCsrView& labelMatrix, const SinglePartition& partition,
                                                  IStatistics& statistics,
                                                  const EqualWeightVector& exampleWeights) const override {
            return std::make_unique<ExampleWiseStratifiedSampling<BinaryCsrView, SinglePartition::const_iterator,
                                                                  EqualWeightVector, BitWeightVector>>(
              rngFactoryPtr_->create(), labelMatrix, partition.cbegin(), partition.cend(), exampleWeights, sampleSize_,
              minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const BinaryCsrView& labelMatrix, const SinglePartition& partition,
                                                  IStatistics& statistics,
                                                  const DenseWeightVector<float32>& exampleWeights) const override {
            return std::make_unique<ExampleWiseStratifiedSampling<
              BinaryCsrView, SinglePartition::const_iterator, DenseWeightVector<float32>, DenseWeightVector<float32>>>(
              rngFactoryPtr_->create(), labelMatrix, partition.cbegin(), partition.cend(), exampleWeights, sampleSize_,
              minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const BinaryCsrView& labelMatrix, BiPartition& partition,
                                                  IStatistics& statistics,
                                                  const EqualWeightVector& exampleWeights) const override {
            return std::make_unique<ExampleWiseStratifiedSampling<BinaryCsrView, BiPartition::const_iterator,
                                                                  EqualWeightVector, BitWeightVector>>(
              rngFactoryPtr_->create(), labelMatrix, partition.first_cbegin(), partition.first_cend(), exampleWeights,
              sampleSize_, minSamples_, maxSamples_);
        }

        std::unique_ptr<IInstanceSampling> create(const BinaryCsrView& labelMatrix, BiPartition& partition,
                                                  IStatistics& statistics,
                                                  const DenseWeightVector<float32>& exampleWeights) const override {
            return std::make_unique<ExampleWiseStratifiedSampling<
              BinaryCsrView, BiPartition::const_iterator, DenseWeightVector<float32>, DenseWeightVector<float32>>>(
              rngFactoryPtr_->create(), labelMatrix, partition.first_cbegin(), partition.first_cend(), exampleWeights,
              sampleSize_, minSamples_, maxSamples_);
        }
};

ExampleWiseStratifiedInstanceSamplingConfig::ExampleWiseStratifiedInstanceSamplingConfig(
  ReadableProperty<RNGConfig> rngConfig)
    : rngConfig_(rngConfig), sampleSize_(0.66f), minSamples_(1), maxSamples_(0) {}

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

uint32 ExampleWiseStratifiedInstanceSamplingConfig::getMinSamples() const {
    return minSamples_;
}

IExampleWiseStratifiedInstanceSamplingConfig& ExampleWiseStratifiedInstanceSamplingConfig::setMinSamples(
  uint32 minSamples) {
    util::assertGreaterOrEqual<uint32>("minSamples", minSamples, 1);
    minSamples_ = minSamples;
    return *this;
}

uint32 ExampleWiseStratifiedInstanceSamplingConfig::getMaxSamples() const {
    return maxSamples_;
}

IExampleWiseStratifiedInstanceSamplingConfig& ExampleWiseStratifiedInstanceSamplingConfig::setMaxSamples(
  uint32 maxSamples) {
    if (maxSamples != 0) util::assertGreaterOrEqual<uint32>("maxSamples", maxSamples, minSamples_);
    maxSamples_ = maxSamples;
    return *this;
}

std::unique_ptr<IClassificationInstanceSamplingFactory>
  ExampleWiseStratifiedInstanceSamplingConfig::createClassificationInstanceSamplingFactory() const {
    return std::make_unique<ExampleWiseStratifiedInstanceSamplingFactory>(rngConfig_.get().createRNGFactory(),
                                                                          sampleSize_, minSamples_, maxSamples_);
}
