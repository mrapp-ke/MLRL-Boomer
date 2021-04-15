#include "common/sampling/instance_sampling_stratified_label_wise.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "common/input/label_matrix_csc.hpp"


/**
 * Implements iterative stratified sampling for selecting a subset of the available training examples, such that for
 * each label the proportion of relevant and irrelevant examples is maintained.
 *
 * @tparam Partition The type of the object that provides access to the indices of the examples that are included in the
 *                   training set
 */
template<class Partition>
class LabelWiseStratifiedSampling final : public IInstanceSubSampling {

    private:

        Partition& partition_;

        std::unique_ptr<CscLabelMatrix> labelMatrixPtr_;

        float32 sampleSize_;

    public:

        /**
         * @param partition         A reference to an object of template type `Partition` that provides access to the
         *                          indices of the examples that are included in the training set
         * @param labelMatrixPtr    An unique pointer to an object of type `CscLabelMatrix` that provides column-wise
         *                          access to the labels of the training examples
         * @param sampleSize        The fraction of examples to be included in the sample (e.g. a value of 0.6
         *                          corresponds to 60 % of the available examples). Must be in (0, 1]
         */
        LabelWiseStratifiedSampling(Partition& partition, std::unique_ptr<CscLabelMatrix> labelMatrixPtr,
                                    float32 sampleSize)
            : partition_(partition), labelMatrixPtr_(std::move(labelMatrixPtr)), sampleSize_(sampleSize) {

        }

        std::unique_ptr<IWeightVector> subSample(RNG& rng) const override {
            // TODO Implement
            return nullptr;
        }

};

LabelWiseStratifiedSamplingFactory::LabelWiseStratifiedSamplingFactory(float32 sampleSize)
    : sampleSize_(sampleSize) {

}

std::unique_ptr<IInstanceSubSampling> LabelWiseStratifiedSamplingFactory::create(
        const CContiguousLabelMatrix& labelMatrix, const SinglePartition& partition) const {
    std::unique_ptr<CscLabelMatrix> labelMatrixPtr = std::make_unique<CscLabelMatrix>(labelMatrix);
    return std::make_unique<LabelWiseStratifiedSampling<const SinglePartition>>(partition, std::move(labelMatrixPtr),
                                                                                sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> LabelWiseStratifiedSamplingFactory::create(
        const CContiguousLabelMatrix& labelMatrix, BiPartition& partition) const {
    std::unique_ptr<CscLabelMatrix> labelMatrixPtr = std::make_unique<CscLabelMatrix>(labelMatrix);
    return std::make_unique<LabelWiseStratifiedSampling<BiPartition>>(partition, std::move(labelMatrixPtr),
                                                                      sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> LabelWiseStratifiedSamplingFactory::create(
        const CsrLabelMatrix& labelMatrix, const SinglePartition& partition) const {
    std::unique_ptr<CscLabelMatrix> labelMatrixPtr = std::make_unique<CscLabelMatrix>(labelMatrix);
    return std::make_unique<LabelWiseStratifiedSampling<const SinglePartition>>(partition, std::move(labelMatrixPtr),
                                                                                sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> LabelWiseStratifiedSamplingFactory::create(
        const CsrLabelMatrix& labelMatrix, BiPartition& partition) const {
    std::unique_ptr<CscLabelMatrix> labelMatrixPtr = std::make_unique<CscLabelMatrix>(labelMatrix);
    return std::make_unique<LabelWiseStratifiedSampling<BiPartition>>(partition, std::move(labelMatrixPtr),
                                                                      sampleSize_);
}
