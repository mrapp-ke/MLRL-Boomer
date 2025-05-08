#include "mlrl/common/sampling/partition_sampling_bi_stratified_example_wise.hpp"

#include "mlrl/common/iterator/iterator_index.hpp"
#include "mlrl/common/sampling/stratified_sampling_example_wise.hpp"
#include "mlrl/common/util/validation.hpp"

/**
 * Allows to use stratified sampling, where distinct label vectors are treated as individual classes, to split the
 * training examples into two mutually exclusive sets that may be used as a training set and a holdout set.
 *
 * @tparam LabelMatrix The type of the label matrix that provides random or row-wise access to the labels of the
 *                     training examples
 */
template<typename LabelMatrix>
class ExampleWiseStratifiedBiPartitionSampling final : public IPartitionSampling {
    private:

        BiPartition partition_;

        const ExampleWiseStratification<LabelMatrix, IndexIterator> stratification_;

    public:

        /**
         * @param rngPtr        An unique pointer to an object of type `RNG` that should be used for generating random
         *                      numbers
         * @param labelMatrix   A reference to an object of template type `LabelMatrix` that provides random or row-wise
         *                      access to the labels of the training examples
         * @param numTraining   The number of examples to be included in the training set
         * @param numHoldout    The number of examples to be included in the holdout set
         */
        ExampleWiseStratifiedBiPartitionSampling(std::unique_ptr<RNG> rngPtr, const LabelMatrix& labelMatrix,
                                                 uint32 numTraining, uint32 numHoldout)
            : partition_(numTraining, numHoldout),
              stratification_(std::move(rngPtr), labelMatrix, IndexIterator(), IndexIterator(labelMatrix.numRows)) {}

        IPartition& partition() override {
            stratification_.sampleBiPartition(partition_);
            return partition_;
        }
};

/**
 * Allows to create objects of the type `IPartitionSampling` that use stratified sampling, where distinct label vectors
 * are treated as individual classes, to split the training examples into two mutually exclusive sets that may be used
 * as a training set and a holdout set.
 */
class ExampleWiseStratifiedBiPartitionSamplingFactory final : public IClassificationPartitionSamplingFactory {
    private:

        const std::unique_ptr<RNGFactory> rngFactoryPtr_;

        const float32 holdoutSetSize_;

    public:

        /**
         * @param rngFactoryPtr     An unique pointer to an object of type `RNGFactory` that allows to create random
         *                          number generators
         * @param holdoutSetSize    The fraction of examples to be included in the holdout set (e.g. a value of 0.6
         *                          corresponds to 60 % of the available examples). Must be in (0, 1)
         */
        ExampleWiseStratifiedBiPartitionSamplingFactory(std::unique_ptr<RNGFactory> rngFactoryPtr,
                                                        float32 holdoutSetSize)
            : rngFactoryPtr_(std::move(rngFactoryPtr)), holdoutSetSize_(holdoutSetSize) {}

        std::unique_ptr<IPartitionSampling> create(const CContiguousView<const uint8>& labelMatrix) const override {
            uint32 numExamples = labelMatrix.numRows;
            uint32 numHoldout = static_cast<uint32>(holdoutSetSize_ * numExamples);
            uint32 numTraining = numExamples - numHoldout;
            return std::make_unique<ExampleWiseStratifiedBiPartitionSampling<CContiguousView<const uint8>>>(
              rngFactoryPtr_->create(), labelMatrix, numTraining, numHoldout);
        }

        std::unique_ptr<IPartitionSampling> create(const BinaryCsrView& labelMatrix) const override {
            uint32 numExamples = labelMatrix.numRows;
            uint32 numHoldout = static_cast<uint32>(holdoutSetSize_ * numExamples);
            uint32 numTraining = numExamples - numHoldout;
            return std::make_unique<ExampleWiseStratifiedBiPartitionSampling<BinaryCsrView>>(
              rngFactoryPtr_->create(), labelMatrix, numTraining, numHoldout);
        }
};

ExampleWiseStratifiedBiPartitionSamplingConfig::ExampleWiseStratifiedBiPartitionSamplingConfig(
  ReadableProperty<RNGConfig> rngConfig)
    : rngConfig_(rngConfig), holdoutSetSize_(0.33f) {}

float32 ExampleWiseStratifiedBiPartitionSamplingConfig::getHoldoutSetSize() const {
    return holdoutSetSize_;
}

IExampleWiseStratifiedBiPartitionSamplingConfig& ExampleWiseStratifiedBiPartitionSamplingConfig::setHoldoutSetSize(
  float32 holdoutSetSize) {
    util::assertGreater<float32>("holdoutSetSize", holdoutSetSize, 0);
    util::assertLess<float32>("holdoutSetSize", holdoutSetSize, 1);
    holdoutSetSize_ = holdoutSetSize;
    return *this;
}

std::unique_ptr<IClassificationPartitionSamplingFactory>
  ExampleWiseStratifiedBiPartitionSamplingConfig::createClassificationPartitionSamplingFactory() const {
    return std::make_unique<ExampleWiseStratifiedBiPartitionSamplingFactory>(rngConfig_.get().createRNGFactory(),
                                                                             holdoutSetSize_);
}
