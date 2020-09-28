#include "sub_sampling.h"
#include <unordered_set>
#include <math.h>


/**
 * Randomly selects `numSamples` out of `numTotal` elements and sets their weights to 1, while the remaining weights are
 * set to 0, by using a set to keep track of the elements that have already been selected. This method is suitable if
 * `numSamples` is much smaller than `numTotal`.
 *
 * @param numTotal      The total number of available elements
 * @param numSamples    The number of weights to be set to 1
 * @param rng           A pointer to an object of type `RNG`, implementing the random number generator to be used
 * @return              A pointer to an object of type `IWeightVector` that provides access to the weights
 */
static inline IWeightVector* sampleWeightsWithoutReplacementViaTrackingSelection(uint32 numTotal, uint32 numSamples,
                                                                                 RNG* rng) {
    DenseVector<uint8>* weights = new DenseVector<uint8>(numTotal);
    std::unordered_set<uint32> selectedIndices;

    for (uint32 i = 0; i < numSamples; i++) {
        bool shouldContinue = true;
        uint32 randomIndex;

        while (shouldContinue) {
            randomIndex = rng->random(0, numTotal);
            shouldContinue = !selectedIndices.insert(randomIndex).second;
        }

        weights->setValue(randomIndex, 1);
    }

    return new DenseWeightVector<uint8>(weights, numSamples);
}

/**
 * Randomly selects `numSamples` out of `numTotal` elements and sets their weights to 1, while the remaining weights are
 * set to 0, by using a pool, i.e., an array, to keep track of the elements that have not been selected yet.
 *
 * @param numTotal      The total number of available elements
 * @param numSamples    The number of weights to be set to 1
 * @param rng           A pointer to an object of type `RNG`, implementing the random number generator to be used
 * @return              A pointer to an object of type `IWeightVector` that provides access to the weights
 */
static inline IWeightVector* sampleWeightsWithoutReplacementViaPool(uint32 numTotal, uint32 numSamples, RNG* rng) {
    DenseVector<uint8>* weights = new DenseVector<uint8>(numTotal);
    uint32 pool[numTotal];

    // Initialize pool...
    for (uint32 i = 0; i < numTotal; i++) {
        pool[i] = i;
    }

    for (uint32 i = 0; i < numSamples; i++) {
        // Randomly select an index that has not been drawn yet...
        uint32 randomIndex = rng->random(0, numTotal - i);
        uint32 j = pool[randomIndex];

        // Set weight at the selected index to 1...
        weights->setValue(j, 1);

        // Move the index at the border to the position of the recently drawn index...
        pool[randomIndex] = pool[numTotal - i - 1];
    }

    return new DenseWeightVector<uint8>(weights, numSamples);
}

/**
 * Randomly selects `numSamples` out of `numTotal` elements and sets their weights to 1, while the remaining weights are
 * set to 0. The method that is used internally is chosen automatically, depending on the ratio `numSamples / numTotal`.
 *
 * @param numTotal      The total number of available elements
 * @param numSamples    The number of weights to be set to 1
 * @param rng           A pointer to an object of type `RNG`, implementing the random number generator to be used
 * @return              A pointer to an object of type `IWeightVector` that provides access to the weights
 *
 */
static inline IWeightVector* sampleWeightsWithoutReplacement(uint32 numTotal, uint32 numSamples, RNG* rng) {
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
 * @param rng           A pointer to an object of type `RNG`, implementing the random number generator to be used
 * @return              A pointer to an object of type `IIndexVector` that provides access to the indices that are
 *                      contained in the sub-sample
 */
static inline IIndexVector* sampleIndicesWithoutReplacementViaTrackingSelection(uint32 numTotal, uint32 numSamples,
                                                                                RNG* rng) {
    DenseIndexVector* indices = new DenseIndexVector(numSamples);
    std::unordered_set<uint32> selectedIndices;

    for (uint32 i = 0; i < numSamples; i++) {
        bool shouldContinue = true;
        uint32 randomIndex;

        while (shouldContinue) {
            randomIndex = rng->random(0, numTotal);
            shouldContinue = !selectedIndices.insert(randomIndex).second;
        }

        indices->setIndex(i, randomIndex);
    }

    return indices;
}

/**
 * Randomly selects `numSamples` out of `numTotal` indices without replacement using a reservoir sampling algorithm.
 * This method is suitable if `numSamples` is almost as large as `numTotal`.
 *
 * @param numTotal      The total number of available indices
 * @param numSamples    The number of indices to be sampled
 * @param rng           A pointer to an object of type `RNG`, implementing the random number generator to be used
 * @return              A pointer to an object of type `IIndexVector` that provides access to the indices that are
 *                      contained in the sub-sample
 */
static inline IIndexVector* sampleIndicesWithoutReplacementViaReservoirSampling(uint32 numTotal, uint32 numSamples,
                                                                                RNG* rng) {
    DenseIndexVector* indices = new DenseIndexVector(numSamples);

    for (uint32 i = 0; i < numSamples; i++) {
        indices->setIndex(i, i);
    }

    for (uint32 i = numSamples; i < numTotal; i++) {
        uint32 randomIndex = rng->random(0, i + 1);

        if (randomIndex < numSamples) {
            indices->setIndex(randomIndex, i);
        }
    }

    return indices;
}

/**
 * Randomly selects `numSamples` out of `numTotal` indices without replacement by first generating a random permutation
 * of the available indices using the Fisher-Yates shuffle and then returning the first `numSamples` indices.
 *
 * @param numTotal      The total number of available indices
 * @param numSamples    The number of indices to be sampled
 * @param rng           A pointer to an object of type `RNG`, implementing the random number generator to be used
 * @return              A pointer to an object of type `IIndexVector` that provides access to the indices that are
 *                      contained in the sub-sample
 */
static inline IIndexVector* sampleIndicesWithoutReplacementViaRandomPermutation(uint32 numTotal, uint32 numSamples,
                                                                                RNG* rng) {
    DenseIndexVector* indices = new DenseIndexVector(numSamples);
    uint32* unusedIndices = new uint32[numTotal - numSamples];

    for (uint32 i = 0; i < numSamples; i++) {
        indices->setIndex(i, i);
    }

    for (uint32 i = numSamples; i < numTotal; i++) {
        unusedIndices[i - numSamples] = i;
    }

    for (uint32 i = 0; i < numTotal - 2; i++) {
        // Swap elements at index i and at a randomly selected index...
        uint32 randomIndex = rng->random(i, numTotal);
        uint32 tmp1 = i < numSamples ? indices->getIndex(i) : unusedIndices[i - numSamples];
        uint32 tmp2;

        if (randomIndex < numSamples) {
            tmp2 = indices->getIndex(randomIndex);
            indices->setIndex(randomIndex, tmp1);
        } else {
            tmp2 = unusedIndices[randomIndex - numSamples];
            unusedIndices[randomIndex - numSamples] = tmp1;
        }

        if (i < numSamples) {
            indices->setIndex(i, tmp2);
        } else {
            unusedIndices[i - numSamples] = tmp2;
        }
    }

    delete[] unusedIndices;
    return indices;
}

/**
 * Randomly selects `numSamples` out of `numTotal` indices without replacement. The method that is used internally is
 * chosen automatically, depending on the ratio `numSamples / numTotal`.
 *
 * @param numTotal      The total number of available indices
 * @param numSamples    The number of indices to be sampled
 * @param rng           A pointer to an object of type `RNG`, implementing the random number generator to be used
 * @return              A pointer to an object of type `IIndexVector` that provides access to the indices that are
 *                      contained in the sub-sample
 */
static inline IIndexVector* sampleIndicesWithoutReplacement(uint32 numTotal, uint32 numSamples, RNG* rng) {
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

template<class T>
DenseWeightVector<T>::DenseWeightVector(DenseVector<T>* weights, uint32 sumOfWeights) {
    weights_ = weights;
    sumOfWeights = sumOfWeights;
}

template<class T>
DenseWeightVector<T>::~DenseWeightVector() {
    delete weights_;
}

template<class T>
uint32 DenseWeightVector<T>::getNumElements() {
    return weights_->getNumElements();
}

template<class T>
bool DenseWeightVector<T>::hasZeroElements() {
    return false;
}

template<class T>
uint32 DenseWeightVector<T>::getValue(uint32 pos) {
    return (uint32) weights_->getValue(pos);
}

template<class T>
uint32 DenseWeightVector<T>::getSumOfWeights() {
    return sumOfWeights_;
}

EqualWeightVector::EqualWeightVector(uint32 numElements) {
    numElements_ = numElements;
}

uint32 EqualWeightVector::getNumElements() {
    return numElements_;
}

bool EqualWeightVector::hasZeroElements() {
    return false;
}

uint32 EqualWeightVector::getValue(uint32 pos) {
    return 1;
}

uint32 EqualWeightVector::getSumOfWeights() {
    return numElements_;
}

BaggingImpl::BaggingImpl(float32 sampleSize) {
    sampleSize_ = sampleSize;
}

IWeightVector* BaggingImpl::subSample(uint32 numExamples, RNG* rng) {
    uint32 numSamples = (uint32) (sampleSize_ * numExamples);
    DenseVector<uint32>* weights = new DenseVector<uint32>(numExamples);

    for (uint32 i = 0; i < numSamples; i++) {
        // Randomly select the index of an example...
        uint32 randomIndex = rng->random(0, numExamples);

        // Update weight at the selected index...
        uint32 weight = weights->getValue(randomIndex);
        weights->setValue(randomIndex, weight + 1);
    }

    return new DenseWeightVector<uint32>(weights, numSamples);
}

RandomInstanceSubsetSelectionImpl::RandomInstanceSubsetSelectionImpl(float32 sampleSize) {
    sampleSize_ = sampleSize;
}

IWeightVector* RandomInstanceSubsetSelectionImpl::subSample(uint32 numExamples, RNG* rng) {
    uint32 numSamples = (uint32) (sampleSize_ * numExamples);
    return sampleWeightsWithoutReplacement(numExamples, numSamples, rng);
}

IWeightVector* NoInstanceSubSamplingImpl::subSample(uint32 numExamples, RNG* rng) {
    return new EqualWeightVector(numExamples);
}

RandomFeatureSubsetSelectionImpl::RandomFeatureSubsetSelectionImpl(float32 sampleSize) {
    sampleSize_ = sampleSize;
}

IIndexVector* RandomFeatureSubsetSelectionImpl::subSample(uint32 numFeatures, RNG* rng) {
    uint32 numSamples;

    if (sampleSize_ > 0) {
            numSamples = (uint32) (sampleSize_ * numFeatures);
    } else {
            numSamples = (uint32) (log2(numFeatures - 1) + 1);
    }

    return sampleIndicesWithoutReplacement(numFeatures, numSamples, rng);
}

IIndexVector* NoFeatureSubSamplingImpl::subSample(uint32 numFeatures, RNG* rng) {
    return new RangeIndexVector(numFeatures);
}

RandomLabelSubsetSelectionImpl::RandomLabelSubsetSelectionImpl(uint32 numSamples) {
    numSamples_ = numSamples;
}

IIndexVector* RandomLabelSubsetSelectionImpl::subSample(uint32 numLabels, RNG* rng) {
    return sampleIndicesWithoutReplacement(numLabels, numSamples_, rng);
}

IIndexVector* NoLabelSubSamplingImpl::subSample(uint32 numLabels, RNG* rng) {
    return new RangeIndexVector(numLabels);
}
