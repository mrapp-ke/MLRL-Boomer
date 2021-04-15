#include "common/sampling/instance_sampling_stratified_label_wise.hpp"
#include "common/sampling/weight_vector_dense.hpp"
#include "common/sampling/partition_bi.hpp"
#include "common/sampling/partition_single.hpp"
#include "common/input/label_matrix_csc.hpp"
#include "common/data/arrays.hpp"
#include <limits>

#define UNSET_WEIGHT 2


static inline void fetchNumExamplesPerLabel(const CscLabelMatrix& labelMatrix, DenseVector<uint32>::iterator iterator) {
    uint32 numCols = labelMatrix.getNumCols();

    for (uint32 i = 0; i < numCols; i++) {
        uint32 numExamples = labelMatrix.column_indices_cend(i) - labelMatrix.column_indices_cbegin(i);
        iterator[i] = numExamples;
    }
}

static inline uint32 getLabelWithFewestExamples(DenseVector<uint32>::const_iterator numExamplesIterator,
                                                uint32 numElements) {
    uint32 minNumExamples = std::numeric_limits<uint32>::max();
    uint32 index = numElements;

    for (uint32 i = 1; i < numElements; i++) {
        uint32 numExamples = numExamplesIterator[i];

        if (numExamples > 0 && numExamples < minNumExamples) {
            minNumExamples = numExamples;
            index = i;
        }
    }

    return index;
}

static inline void updateNumExamplesPerLabel(const CContiguousLabelMatrix& labelMatrix, uint32 exampleIndex,
                                             DenseVector<uint32>::iterator numExamplesIterator) {
    CContiguousLabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
    uint32 numLabels = labelMatrix.getNumCols();

    for (uint32 i = 0; i < numLabels; i++) {
        if (labelIterator[i]) {
            numExamplesIterator[i]--;
        }
    }
}

static inline void updateNumExamplesPerLabel(const CsrLabelMatrix& labelMatrix, uint32 exampleIndex,
                                             DenseVector<uint32>::iterator numExamplesIterator) {
    CsrLabelMatrix::index_const_iterator indexIterator = labelMatrix.row_indices_cbegin(exampleIndex);
    uint32 numLabels = labelMatrix.row_indices_cend(exampleIndex) - indexIterator;

    for (uint32 i = 0; i < numLabels; i++) {
        uint32 labelIndex = indexIterator[i];
        numExamplesIterator[labelIndex]--;
    }
}

/**
 * Implements iterative stratified sampling for selecting a subset of the available training examples, such that for
 * each label the proportion of relevant and irrelevant examples is maintained.
 *
 * @tparam Partition    The type of the object that provides access to the indices of the examples that are included in
 *                      the training set
 * @tparam LabelMatrix  The type of the label matrix that provides random or row-wise access to the labels of the
 *                      training examples
 */
template<class Partition, class LabelMatrix>
class LabelWiseStratifiedSampling final : public IInstanceSubSampling {

    private:

        Partition& partition_;

        const LabelMatrix& labelMatrix_;

        CscLabelMatrix cscLabelMatrix_;

        float32 sampleSize_;

        DenseVector<uint32> numExamplesVector_;

    public:

        /**
         * @param partition     A reference to an object of template type `Partition` that provides access to the
         *                      indices of the examples that are included in the training set
         * @param labelMatrix   A reference to an object of template type `LabelMatrix` that provides random or row-wise
         *                      access to the labels of the training examples
         * @param sampleSize    The fraction of examples to be included in the sample (e.g. a value of 0.6 corresponds
         *                      to 60 % of the available examples). Must be in (0, 1]
         */
        LabelWiseStratifiedSampling(Partition& partition, const LabelMatrix& labelMatrix, float32 sampleSize)
            : partition_(partition), labelMatrix_(labelMatrix), cscLabelMatrix_(CscLabelMatrix(labelMatrix)),
              sampleSize_(sampleSize), numExamplesVector_(DenseVector<uint32>(labelMatrix.getNumCols())) {

        }

        std::unique_ptr<IWeightVector> subSample(RNG& rng) override {
            // Create a vector to store the weights of individual examples...
            uint32 numExamples = partition_.getNumElements();
            std::unique_ptr<DenseWeightVector<uint32>> weightVectorPtr =
                std::make_unique<DenseWeightVector<uint32>>(numExamples);
            DenseWeightVector<uint32>::iterator weightIterator = weightVectorPtr->begin();

            // Initialize the weights of all examples with the value `UNSET_WEIGHT`, which allows to identify examples
            // for which no weight has been set yet...
            setArrayToValue<uint32>(weightIterator, numExamples, UNSET_WEIGHT);

            // Determine the number of examples that are associated with individual labels...
            DenseVector<uint32>::iterator numExamplesIterator = numExamplesVector_.begin();
            fetchNumExamplesPerLabel(cscLabelMatrix_, numExamplesIterator);
            uint32 numLabels = numExamplesVector_.getNumElements();

            // For each label, assign a weight to the examples that are associated with the label, if no weight has been
            // set yet. Labels with few examples are processed first...
            uint32 numNonZeroWeights = 0;
            uint32 labelIndex;

            while ((labelIndex = getLabelWithFewestExamples(numExamplesIterator, numLabels)) < numLabels) {
                CscLabelMatrix::index_iterator indexIterator = cscLabelMatrix_.column_indices_begin(labelIndex);
                uint32 numExamples = cscLabelMatrix_.column_indices_end(labelIndex) - indexIterator;
                uint32 numSamples = (uint32) (sampleSize_ * numExamplesIterator[labelIndex]);
                numNonZeroWeights += numSamples;
                uint32 i;

                // Use the Fisher-Yates shuffle to randomly draw `numSamples` examples for which no weight has been set
                // yet and set their weight to 1...
                for (i = 0; i < numExamples && numSamples > 0; i++) {
                    uint32 randomIndex = rng.random(i, numExamples);
                    uint32 exampleIndex = indexIterator[randomIndex];
                    indexIterator[randomIndex] = indexIterator[i];
                    indexIterator[i] = exampleIndex;

                    if (weightIterator[exampleIndex] == UNSET_WEIGHT) {
                        weightIterator[exampleIndex] = 1;
                        updateNumExamplesPerLabel(labelMatrix_, exampleIndex, numExamplesIterator);
                        numSamples--;
                    }
                }

                // Set the weights of the remaining examples to 0, if no weight has been set yet...
                for (i = i + 1; i < numExamples; i++) {
                    uint32 exampleIndex = indexIterator[i];

                    if (weightIterator[exampleIndex] == UNSET_WEIGHT) {
                        weightIterator[exampleIndex] = 0;
                        updateNumExamplesPerLabel(labelMatrix_, exampleIndex, numExamplesIterator);
                    }
                }
            }

            weightVectorPtr->setNumNonZeroWeights(numNonZeroWeights);
            return weightVectorPtr;
        }

};

LabelWiseStratifiedSamplingFactory::LabelWiseStratifiedSamplingFactory(float32 sampleSize)
    : sampleSize_(sampleSize) {

}

std::unique_ptr<IInstanceSubSampling> LabelWiseStratifiedSamplingFactory::create(
        const CContiguousLabelMatrix& labelMatrix, const SinglePartition& partition) const {
    return std::make_unique<LabelWiseStratifiedSampling<const SinglePartition, CContiguousLabelMatrix>>(partition,
                                                                                                        labelMatrix,
                                                                                                        sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> LabelWiseStratifiedSamplingFactory::create(
        const CContiguousLabelMatrix& labelMatrix, BiPartition& partition) const {
    return std::make_unique<LabelWiseStratifiedSampling<BiPartition, CContiguousLabelMatrix>>(partition, labelMatrix,
                                                                                              sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> LabelWiseStratifiedSamplingFactory::create(
        const CsrLabelMatrix& labelMatrix, const SinglePartition& partition) const {
    return std::make_unique<LabelWiseStratifiedSampling<const SinglePartition, CsrLabelMatrix>>(partition, labelMatrix,
                                                                                                sampleSize_);
}

std::unique_ptr<IInstanceSubSampling> LabelWiseStratifiedSamplingFactory::create(
        const CsrLabelMatrix& labelMatrix, BiPartition& partition) const {
    return std::make_unique<LabelWiseStratifiedSampling<BiPartition, CsrLabelMatrix>>(partition, labelMatrix,
                                                                                      sampleSize_);
}
