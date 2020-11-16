#include "sub_sampling.h"
#include "data/vector_dense.h"
#include "indices/index_vector_full.h"
#include "indices/index_vector_partial.h"
#include <unordered_set>
#include <math.h>


/**
 * An one-dimensional vector that provides random access to a fixed number of weights stored in a C-contiguous array.
 */
class DenseWeightVector : public IWeightVector {

    private:

        DenseVector<uint32> vector_;

        uint32 sumOfWeights_;

    public:

        /**
         * @param numElements   The number of elements in the vector. Must be at least 1
         * @param sumOfWeights  The sum of the weights in the vector
         */
        DenseWeightVector(uint32 numElements, uint32 sumOfWeights)
            : vector_(DenseVector<uint32>(numElements, true)), sumOfWeights_(sumOfWeights) {

        }

        typedef DenseVector<uint32>::iterator iterator;

        typedef DenseVector<uint32>::const_iterator const_iterator;

        /**
         * Returns an `iterator` to the beginning of the vector.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin() {
            return vector_.begin();
        }

        /**
         * Returns an `iterator` to the end of the vector.
         *
         * @return An `iterator` to the end
         */
        iterator end() {
            return vector_.end();
        }

        /**
         * Returns a `const_iterator` to the beginning of the vector.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const {
            return vector_.cbegin();
        }

        /**
         * Returns a `const_iterator` to the end of the vector.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const {
            return vector_.cend();
        }

        bool hasZeroWeights() const override {
            return true;
        }

        uint32 getWeight(uint32 pos) const override {
            return vector_.getValue(pos);
        }

        uint32 getSumOfWeights() const override {
            return sumOfWeights_;
        }

};

/**
 * An one-dimensional that provides random access to a fixed number of equal weights.
 */
class EqualWeightVector : public IWeightVector {

    private:

        uint32 numElements_;

    public:

        /**
         * @param numTotalElements The number of elements in the vector. Must be at least 1
         */
        EqualWeightVector(uint32 numElements)
            : numElements_(numElements) {

        }

        bool hasZeroWeights() const override {
            return false;
        }

        uint32 getWeight(uint32 pos) const override {
            return 1;
        }

        uint32 getSumOfWeights() const override {
            return numElements_;
        }

};

/**
 * Randomly selects `numSamples` out of `numTotal` elements and sets their weights to 1, while the remaining weights are
 * set to 0, by using a set to keep track of the elements that have already been selected. This method is suitable if
 * `numSamples` is much smaller than `numTotal`.
 *
 * @param numTotal      The total number of available elements
 * @param numSamples    The number of weights to be set to 1
 * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be used
 * @return              An unique pointer to an object of type `IWeightVector` that provides access to the weights
 */
static inline std::unique_ptr<IWeightVector> sampleWeightsWithoutReplacementViaTrackingSelection(uint32 numTotal,
                                                                                                 uint32 numSamples,
                                                                                                 RNG& rng) {
    std::unique_ptr<DenseWeightVector> weightVectorPtr = std::make_unique<DenseWeightVector>(numTotal, numSamples);
    DenseWeightVector::iterator iterator = weightVectorPtr->begin();
    std::unordered_set<uint32> selectedIndices;

    for (uint32 i = 0; i < numSamples; i++) {
        bool shouldContinue = true;
        uint32 randomIndex;

        while (shouldContinue) {
            randomIndex = rng.random(0, numTotal);
            shouldContinue = !selectedIndices.insert(randomIndex).second;
        }

        iterator[randomIndex] = 1;
    }

    return weightVectorPtr;
}

/**
 * Randomly selects `numSamples` out of `numTotal` elements and sets their weights to 1, while the remaining weights are
 * set to 0, by using a pool, i.e., an array, to keep track of the elements that have not been selected yet.
 *
 * @param numTotal      The total number of available elements
 * @param numSamples    The number of weights to be set to 1
 * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be used
 * @return              An unique pointer to an object of type `IWeightVector` that provides access to the weights
 */
static inline std::unique_ptr<IWeightVector> sampleWeightsWithoutReplacementViaPool(uint32 numTotal, uint32 numSamples,
                                                                                    RNG& rng) {
    std::unique_ptr<DenseWeightVector> weightVectorPtr = std::make_unique<DenseWeightVector>(numTotal, numSamples);
    DenseWeightVector::iterator iterator = weightVectorPtr->begin();
    uint32 pool[numTotal];

    // Initialize pool...
    for (uint32 i = 0; i < numTotal; i++) {
        pool[i] = i;
    }

    for (uint32 i = 0; i < numSamples; i++) {
        // Randomly select an index that has not been drawn yet...
        uint32 randomIndex = rng.random(0, numTotal - i);
        uint32 j = pool[randomIndex];

        // Set weight at the selected index to 1...
        iterator[j] = 1;

        // Move the index at the border to the position of the recently drawn index...
        pool[randomIndex] = pool[numTotal - i - 1];
    }

    return weightVectorPtr;
}

/**
 * Randomly selects `numSamples` out of `numTotal` elements and sets their weights to 1, while the remaining weights are
 * set to 0. The method that is used internally is chosen automatically, depending on the ratio `numSamples / numTotal`.
 *
 * @param numTotal      The total number of available elements
 * @param numSamples    The number of weights to be set to 1
 * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be used
 * @return              An unique pointer to an object of type `IWeightVector` that provides access to the weights
 *
 */
static inline std::unique_ptr<IWeightVector> sampleWeightsWithoutReplacement(uint32 numTotal, uint32 numSamples,
                                                                             RNG& rng) {
    float64 ratio = numTotal > 0 ? ((float64) numSamples) / ((float64) numTotal) : 1;

    if (ratio < 0.06) {
        // For very small ratios use tracking selection
        return sampleWeightsWithoutReplacementViaTrackingSelection(numTotal, numSamples, rng);
    } else {
        // Otherwise, use a pool as the default method
        return sampleWeightsWithoutReplacementViaPool(numTotal, numSamples, rng);
    }
}

/**
 * Randomly selects `numSamples` out of `numTotal` indices without replacement by using a set to keep track of the
 * indices that have already been selected. This method is suitable if `numSamples` is much smaller than `numTotal`
 *
 * @param numTotal      The total number of available indices
 * @param numSamples    The number of indices to be sampled
 * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be used
 * @return              An unique pointer to an object of type `IIndexVector` that provides access to the indices that
 *                      are contained in the sub-sample
 */
static inline std::unique_ptr<IIndexVector> sampleIndicesWithoutReplacementViaTrackingSelection(uint32 numTotal,
                                                                                                uint32 numSamples,
                                                                                                RNG& rng) {
    std::unique_ptr<PartialIndexVector> indexVectorPtr = std::make_unique<PartialIndexVector>(numSamples);
    PartialIndexVector::iterator iterator = indexVectorPtr->begin();
    std::unordered_set<uint32> selectedIndices;

    for (uint32 i = 0; i < numSamples; i++) {
        bool shouldContinue = true;
        uint32 randomIndex;

        while (shouldContinue) {
            randomIndex = rng.random(0, numTotal);
            shouldContinue = !selectedIndices.insert(randomIndex).second;
        }

        iterator[i] = randomIndex;
    }

    return indexVectorPtr;;
}

/**
 * Randomly selects `numSamples` out of `numTotal` indices without replacement using a reservoir sampling algorithm.
 * This method is suitable if `numSamples` is almost as large as `numTotal`.
 *
 * @param numTotal      The total number of available indices
 * @param numSamples    The number of indices to be sampled
 * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be used
 * @return              A pointer to an object of type `IIndexVector` that provides access to the indices that are
 *                      contained in the sub-sample
 */
static inline std::unique_ptr<IIndexVector> sampleIndicesWithoutReplacementViaReservoirSampling(uint32 numTotal,
                                                                                                uint32 numSamples,
                                                                                                RNG& rng) {
    std::unique_ptr<PartialIndexVector> indexVectorPtr = std::make_unique<PartialIndexVector>(numSamples);
    PartialIndexVector::iterator iterator = indexVectorPtr->begin();

    for (uint32 i = 0; i < numSamples; i++) {
        iterator[i] = i;
    }

    for (uint32 i = numSamples; i < numTotal; i++) {
        uint32 randomIndex = rng.random(0, i + 1);

        if (randomIndex < numSamples) {
            iterator[randomIndex] = i;
        }
    }

    return indexVectorPtr;
}

/**
 * Randomly selects `numSamples` out of `numTotal` indices without replacement by first generating a random permutation
 * of the available indices using the Fisher-Yates shuffle and then returning the first `numSamples` indices.
 *
 * @param numTotal      The total number of available indices
 * @param numSamples    The number of indices to be sampled
 * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be used
 * @return              An unique pointer to an object of type `IIndexVector` that provides access to the indices that
 *                      are contained in the sub-sample
 */
static inline std::unique_ptr<IIndexVector> sampleIndicesWithoutReplacementViaRandomPermutation(uint32 numTotal,
                                                                                                uint32 numSamples,
                                                                                                RNG& rng) {
    std::unique_ptr<PartialIndexVector> indexVectorPtr = std::make_unique<PartialIndexVector>(numSamples);
    PartialIndexVector::iterator iterator = indexVectorPtr->begin();
    uint32 unusedIndices[numTotal - numSamples];

    for (uint32 i = 0; i < numSamples; i++) {
        iterator[i] = i;
    }

    for (uint32 i = numSamples; i < numTotal; i++) {
        unusedIndices[i - numSamples] = i;
    }

    for (uint32 i = 0; i < numTotal - 2; i++) {
        // Swap elements at index i and at a randomly selected index...
        uint32 randomIndex = rng.random(i, numTotal);
        uint32 tmp1 = i < numSamples ? iterator[i] : unusedIndices[i - numSamples];
        uint32 tmp2;

        if (randomIndex < numSamples) {
            tmp2 = iterator[randomIndex];
            iterator[randomIndex] = tmp1;
        } else {
            tmp2 = unusedIndices[randomIndex - numSamples];
            unusedIndices[randomIndex - numSamples] = tmp1;
        }

        if (i < numSamples) {
            iterator[i] = tmp2;
        } else {
            unusedIndices[i - numSamples] = tmp2;
        }
    }

    return indexVectorPtr;
}

/**
 * Randomly selects `numSamples` out of `numTotal` indices without replacement. The method that is used internally is
 * chosen automatically, depending on the ratio `numSamples / numTotal`.
 *
 * @param numTotal      The total number of available indices
 * @param numSamples    The number of indices to be sampled
 * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be used
 * @return              An unique pointer to an object of type `IIndexVector` that provides access to the indices that
 *                      are contained in the sub-sample
 */
static inline std::unique_ptr<IIndexVector> sampleIndicesWithoutReplacement(uint32 numTotal, uint32 numSamples,
                                                                            RNG& rng) {
    float64 ratio = numTotal > 0 ? ((float64) numSamples) / ((float64) numTotal) : 1;

    // The thresholds for choosing a suitable method are based on empirical experiments
    if (ratio < 0.06) {
        // For very small ratios use tracking selection
        return sampleIndicesWithoutReplacementViaTrackingSelection(numTotal, numSamples, rng);
    } else if (ratio > 0.5) {
        // For large ratios use reservoir sampling
        return sampleIndicesWithoutReplacementViaReservoirSampling(numTotal, numSamples, rng);
    } else {
        // Otherwise, use random permutation as the default method
        return sampleIndicesWithoutReplacementViaRandomPermutation(numTotal, numSamples, rng);
    }
}

BaggingImpl::BaggingImpl(float32 sampleSize)
    : sampleSize_(sampleSize) {

}

std::unique_ptr<IWeightVector> BaggingImpl::subSample(uint32 numExamples, RNG& rng) const {
    uint32 numSamples = (uint32) (sampleSize_ * numExamples);
    std::unique_ptr<DenseWeightVector> weightVectorPtr = std::make_unique<DenseWeightVector>(numExamples, numSamples);
    DenseWeightVector::iterator iterator = weightVectorPtr->begin();

    for (uint32 i = 0; i < numSamples; i++) {
        // Randomly select the index of an example...
        uint32 randomIndex = rng.random(0, numExamples);

        // Update weight at the selected index...
        iterator[randomIndex] += 1;
    }

    return weightVectorPtr;
}

RandomInstanceSubsetSelectionImpl::RandomInstanceSubsetSelectionImpl(float32 sampleSize)
    : sampleSize_(sampleSize) {

}

std::unique_ptr<IWeightVector> RandomInstanceSubsetSelectionImpl::subSample(uint32 numExamples, RNG& rng) const {
    uint32 numSamples = (uint32) (sampleSize_ * numExamples);
    return sampleWeightsWithoutReplacement(numExamples, numSamples, rng);
}

std::unique_ptr<IWeightVector> NoInstanceSubSamplingImpl::subSample(uint32 numExamples, RNG& rng) const {
    return std::make_unique<EqualWeightVector>(numExamples);
}

RandomFeatureSubsetSelectionImpl::RandomFeatureSubsetSelectionImpl(float32 sampleSize)
    : sampleSize_(sampleSize) {

}

std::unique_ptr<IIndexVector> RandomFeatureSubsetSelectionImpl::subSample(uint32 numFeatures, RNG& rng) const {
    uint32 numSamples;

    if (sampleSize_ > 0) {
            numSamples = (uint32) (sampleSize_ * numFeatures);
    } else {
            numSamples = (uint32) (log2(numFeatures - 1) + 1);
    }

    return sampleIndicesWithoutReplacement(numFeatures, numSamples, rng);
}

std::unique_ptr<IIndexVector> NoFeatureSubSamplingImpl::subSample(uint32 numFeatures, RNG& rng) const {
    return std::make_unique<FullIndexVector>(numFeatures);
}

RandomLabelSubsetSelectionImpl::RandomLabelSubsetSelectionImpl(uint32 numSamples)
    : numSamples_(numSamples) {

}

std::unique_ptr<IIndexVector> RandomLabelSubsetSelectionImpl::subSample(uint32 numLabels, RNG& rng) const {
    return sampleIndicesWithoutReplacement(numLabels, numSamples_, rng);
}

std::unique_ptr<IIndexVector> NoLabelSubSamplingImpl::subSample(uint32 numLabels, RNG& rng) const {
    return std::make_unique<FullIndexVector>(numLabels);
}
