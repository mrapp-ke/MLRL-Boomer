#include "common/sampling/instance_sampling_stratified_label_wise.hpp"

#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "common/sampling/stratified_sampling_label_wise.hpp"
#include "common/util/validation.hpp"

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
class LabelWiseStratifiedSampling final : public IInstanceSampling {
    private:

        const float32 sampleSize_;

        BitWeightVector weightVector_;

        const LabelWiseStratification<LabelMatrix, IndexIterator> stratification_;

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
        LabelWiseStratifiedSampling(const LabelMatrix& labelMatrix, IndexIterator indicesBegin,
                                    IndexIterator indicesEnd, float32 sampleSize)
            : sampleSize_(sampleSize),
              weightVector_(BitWeightVector(labelMatrix.getNumRows(),
                                            (uint32) (indicesEnd - indicesBegin) < labelMatrix.getNumRows())),
              stratification_(
                LabelWiseStratification<LabelMatrix, IndexIterator>(labelMatrix, indicesBegin, indicesEnd)) {}

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
class LabelWiseStratifiedInstanceSamplingFactory final : public IInstanceSamplingFactory {
    private:

        const float32 sampleSize_;

    public:

        /**
         * @param sampleSize The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds to
         *                   60 % of the available examples). Must be in (0, 1]
         */
        LabelWiseStratifiedInstanceSamplingFactory(float32 sampleSize) : sampleSize_(sampleSize) {}

        std::unique_ptr<IInstanceSampling> create(const CContiguousLabelMatrix& labelMatrix,
                                                  const SinglePartition& partition,
                                                  IStatistics& statistics) const override {
            return std::make_unique<
              LabelWiseStratifiedSampling<CContiguousLabelMatrix, SinglePartition::const_iterator>>(
              labelMatrix, partition.cbegin(), partition.cend(), sampleSize_);
        }

        std::unique_ptr<IInstanceSampling> create(const CContiguousLabelMatrix& labelMatrix, BiPartition& partition,
                                                  IStatistics& statistics) const override {
            return std::make_unique<LabelWiseStratifiedSampling<CContiguousLabelMatrix, BiPartition::const_iterator>>(
              labelMatrix, partition.first_cbegin(), partition.first_cend(), sampleSize_);
        }

        std::unique_ptr<IInstanceSampling> create(const CsrLabelMatrix& labelMatrix, const SinglePartition& partition,
                                                  IStatistics& statistics) const override {
            return std::make_unique<LabelWiseStratifiedSampling<CsrLabelMatrix, SinglePartition::const_iterator>>(
              labelMatrix, partition.cbegin(), partition.cend(), sampleSize_);
        }

        std::unique_ptr<IInstanceSampling> create(const CsrLabelMatrix& labelMatrix, BiPartition& partition,
                                                  IStatistics& statistics) const override {
            return std::make_unique<LabelWiseStratifiedSampling<CsrLabelMatrix, BiPartition::const_iterator>>(
              labelMatrix, partition.first_cbegin(), partition.first_cend(), sampleSize_);
        }
};

LabelWiseStratifiedInstanceSamplingConfig::LabelWiseStratifiedInstanceSamplingConfig() : sampleSize_(0.66f) {}

float32 LabelWiseStratifiedInstanceSamplingConfig::getSampleSize() const {
    return sampleSize_;
}

ILabelWiseStratifiedInstanceSamplingConfig& LabelWiseStratifiedInstanceSamplingConfig::setSampleSize(
  float32 sampleSize) {
    assertGreater<float32>("sampleSize", sampleSize, 0);
    assertLess<float32>("sampleSize", sampleSize, 1);
    sampleSize_ = sampleSize;
    return *this;
}

std::unique_ptr<IInstanceSamplingFactory> LabelWiseStratifiedInstanceSamplingConfig::createInstanceSamplingFactory()
  const {
    return std::make_unique<LabelWiseStratifiedInstanceSamplingFactory>(sampleSize_);
}
