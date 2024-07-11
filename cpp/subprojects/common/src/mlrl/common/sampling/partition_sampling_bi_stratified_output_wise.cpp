#include "mlrl/common/sampling/partition_sampling_bi_stratified_output_wise.hpp"

#include "mlrl/common/iterator/index_iterator.hpp"
#include "mlrl/common/sampling/stratified_sampling_output_wise.hpp"
#include "mlrl/common/util/validation.hpp"

/**
 * Allows to use stratified sampling to split the training examples into two mutually exclusive sets that may be used as
 * a training set and a holdout set, such that for each label the proportion of relevant and irrelevant examples is
 * maintained.
 *
 * @tparam LabelMatrix The type of the label matrix that provides random or row-wise access to the labels of the
 *                     training examples
 */
template<typename LabelMatrix>
class OutputWiseStratifiedBiPartitionSampling final : public IPartitionSampling {
    private:

        BiPartition partition_;

        LabelWiseStratification<LabelMatrix, IndexIterator> stratification_;

    public:

        /**
         * @param labelMatrix   A reference to an object of template type `LabelMatrix` that provides random or row-wise
         *                      access to the labels of the training examples
         * @param numTraining   The number of examples to be included in the training set
         * @param numHoldout    The number of examples to be included in the holdout set
         */
        OutputWiseStratifiedBiPartitionSampling(const LabelMatrix& labelMatrix, uint32 numTraining, uint32 numHoldout)
            : partition_(numTraining, numHoldout),
              stratification_(labelMatrix, IndexIterator(), IndexIterator(labelMatrix.numRows)) {}

        IPartition& partition(RNG& rng) override {
            stratification_.sampleBiPartition(partition_, rng);
            return partition_;
        }
};

/**
 * Allows to create objects of the type `IPartitionSampling` that use stratified sampling to split the training examples
 * into two mutually exclusive sets that may be used as a training set and a holdout set, such that for each label the
 * proportion of relevant and irrelevant examples is maintained.
 */
class OutputWiseStratifiedBiPartitionSamplingFactory final : public IClassificationPartitionSamplingFactory {
    private:

        const float32 holdoutSetSize_;

    public:

        /**
         * @param holdoutSetSize The fraction of examples to be included in the holdout set (e.g. a value of 0.6
         *                       corresponds to 60 % of the available examples). Must be in (0, 1)
         */
        OutputWiseStratifiedBiPartitionSamplingFactory(float32 holdoutSetSize) : holdoutSetSize_(holdoutSetSize) {}

        std::unique_ptr<IPartitionSampling> create(const CContiguousView<const uint8>& labelMatrix) const override {
            uint32 numExamples = labelMatrix.numRows;
            uint32 numHoldout = static_cast<uint32>(holdoutSetSize_ * numExamples);
            uint32 numTraining = numExamples - numHoldout;
            return std::make_unique<OutputWiseStratifiedBiPartitionSampling<CContiguousView<const uint8>>>(
              labelMatrix, numTraining, numHoldout);
        }

        std::unique_ptr<IPartitionSampling> create(const BinaryCsrView& labelMatrix) const override {
            uint32 numExamples = labelMatrix.numRows;
            uint32 numHoldout = static_cast<uint32>(holdoutSetSize_ * numExamples);
            uint32 numTraining = numExamples - numHoldout;
            return std::make_unique<OutputWiseStratifiedBiPartitionSampling<BinaryCsrView>>(labelMatrix, numTraining,
                                                                                            numHoldout);
        }
};

OutputWiseStratifiedBiPartitionSamplingConfig::OutputWiseStratifiedBiPartitionSamplingConfig()
    : holdoutSetSize_(0.33f) {}

float32 OutputWiseStratifiedBiPartitionSamplingConfig::getHoldoutSetSize() const {
    return holdoutSetSize_;
}

IOutputWiseStratifiedBiPartitionSamplingConfig& OutputWiseStratifiedBiPartitionSamplingConfig::setHoldoutSetSize(
  float32 holdoutSetSize) {
    assertGreater<float32>("holdoutSetSize", holdoutSetSize, 0);
    assertLess<float32>("holdoutSetSize", holdoutSetSize, 1);
    holdoutSetSize_ = holdoutSetSize;
    return *this;
}

std::unique_ptr<IClassificationPartitionSamplingFactory>
  OutputWiseStratifiedBiPartitionSamplingConfig::createClassificationPartitionSamplingFactory() const {
    return std::make_unique<OutputWiseStratifiedBiPartitionSamplingFactory>(holdoutSetSize_);
}
