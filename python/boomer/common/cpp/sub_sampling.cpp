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
 * @return              An unique pointer to an object of type `IWeightVector` that provides access to the weights
 */
static inline std::unique_ptr<IWeightVector> sampleWeightsWithoutReplacementViaTrackingSelection(uint32 numTotal,
                                                                                                 uint32 numSamples,
                                                                                                 RNG* rng) {
    uint8* weights = new uint8[numTotal]{0};
    std::unordered_set<uint32> selectedIndices;

    for (uint32 i = 0; i < numSamples; i++) {
        bool shouldContinue = true;
        uint32 randomIndex;

        while (shouldContinue) {
            randomIndex = rng->random(0, numTotal);
            shouldContinue = !selectedIndices.insert(randomIndex).second;
        }

        weights[randomIndex] = 1;
    }

    return std::make_unique<DenseWeightVector<uint8>>(weights, numTotal, numSamples);
}

/**
 * Randomly selects `numSamples` out of `numTotal` elements and sets their weights to 1, while the remaining weights are
 * set to 0, by using a pool, i.e., an array, to keep track of the elements that have not been selected yet.
 *
 * @param numTotal      The total number of available elements
 * @param numSamples    The number of weights to be set to 1
 * @param rng           A pointer to an object of type `RNG`, implementing the random number generator to be used
 * @return              An unique pointer to an object of type `IWeightVector` that provides access to the weights
 */
static inline std::unique_ptr<IWeightVector> sampleWeightsWithoutReplacementViaPool(uint32 numTotal, uint32 numSamples,
                                                                                    RNG* rng) {
    uint8* weights = new uint8[numTotal]{0};
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
        weights[j] = 1;

        // Move the index at the border to the position of the recently drawn index...
        pool[randomIndex] = pool[numTotal - i - 1];
    }

    return std::make_unique<DenseWeightVector<uint8>>(weights, numTotal, numSamples);
}

/**
 * Randomly selects `numSamples` out of `numTotal` elements and sets their weights to 1, while the remaining weights are
 * set to 0. The method that is used internally is chosen automatically, depending on the ratio `numSamples / numTotal`.
 *
 * @param numTotal      The total number of available elements
 * @param numSamples    The number of weights to be set to 1
 * @param rng           A pointer to an object of type `RNG`, implementing the random number generator to be used
 * @return              An unique pointer to an object of type `IWeightVector` that provides access to the weights
 *
 */
static inline std::unique_ptr<IWeightVector> sampleWeightsWithoutReplacement(uint32 numTotal, uint32 numSamples,
                                                                             RNG* rng) {
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
 * @return              An unique pointer to an object of type `IIndexVector` that provides access to the indices that
 *                      are contained in the sub-sample
 */
static inline std::unique_ptr<IIndexVector> sampleIndicesWithoutReplacementViaTrackingSelection(uint32 numTotal,
                                                                                                uint32 numSamples,
                                                                                                RNG* rng) {
    uint32* indices = new uint32[numSamples];
    std::unordered_set<uint32> selectedIndices;

    for (uint32 i = 0; i < numSamples; i++) {
        bool shouldContinue = true;
        uint32 randomIndex;

        while (shouldContinue) {
            randomIndex = rng->random(0, numTotal);
            shouldContinue = !selectedIndices.insert(randomIndex).second;
        }

        indices[i] = randomIndex;
    }

    return std::make_unique<DenseIndexVector>(indices, numSamples);
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
static inline std::unique_ptr<IIndexVector> sampleIndicesWithoutReplacementViaReservoirSampling(uint32 numTotal,
                                                                                                uint32 numSamples,
                                                                                                RNG* rng) {
    uint32* indices = new uint32[numSamples];

    for (uint32 i = 0; i < numSamples; i++) {
        indices[i] = i;
    }

    for (uint32 i = numSamples; i < numTotal; i++) {
        uint32 randomIndex = rng->random(0, i + 1);

        if (randomIndex < numSamples) {
            indices[randomIndex] = i;
        }
    }

    return std::make_unique<DenseIndexVector>(indices, numSamples);
}

/**
 * Randomly selects `numSamples` out of `numTotal` indices without replacement by first generating a random permutation
 * of the available indices using the Fisher-Yates shuffle and then returning the first `numSamples` indices.
 *
 * @param numTotal      The total number of available indices
 * @param numSamples    The number of indices to be sampled
 * @param rng           A pointer to an object of type `RNG`, implementing the random number generator to be used
 * @return              An unique pointer to an object of type `IIndexVector` that provides access to the indices that
 *                      are contained in the sub-sample
 */
static inline std::unique_ptr<IIndexVector> sampleIndicesWithoutReplacementViaRandomPermutation(uint32 numTotal,
                                                                                                uint32 numSamples,
                                                                                                RNG* rng) {
    uint32* indices = new uint32[numSamples];
    uint32* unusedIndices = new uint32[numTotal - numSamples];

    for (uint32 i = 0; i < numSamples; i++) {
        indices[i] = i;
    }

    for (uint32 i = numSamples; i < numTotal; i++) {
        unusedIndices[i - numSamples] = i;
    }

    for (uint32 i = 0; i < numTotal - 2; i++) {
        // Swap elements at index i and at a randomly selected index...
        uint32 randomIndex = rng->random(i, numTotal);
        uint32 tmp1 = i < numSamples ? indices[i] : unusedIndices[i - numSamples];
        uint32 tmp2;

        if (randomIndex < numSamples) {
            tmp2 = indices[randomIndex];
            indices[randomIndex] = tmp1;
        } else {
            tmp2 = unusedIndices[randomIndex - numSamples];
            unusedIndices[randomIndex - numSamples] = tmp1;
        }

        if (i < numSamples) {
            indices[i] = tmp2;
        } else {
            unusedIndices[i - numSamples] = tmp2;
        }
    }

    delete[] unusedIndices;
    return std::make_unique<DenseIndexVector>(indices, numSamples);
}

/**
 * Randomly selects `numSamples` out of `numTotal` indices without replacement. The method that is used internally is
 * chosen automatically, depending on the ratio `numSamples / numTotal`.
 *
 * @param numTotal      The total number of available indices
 * @param numSamples    The number of indices to be sampled
 * @param rng           A pointer to an object of type `RNG`, implementing the random number generator to be used
 * @return              An unique pointer to an object of type `IIndexVector` that provides access to the indices that
 *                      are contained in the sub-sample
 */
static inline std::unique_ptr<IIndexVector> sampleIndicesWithoutReplacement(uint32 numTotal, uint32 numSamples,
                                                                            RNG* rng) {
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
DenseWeightVector<T>::DenseWeightVector(T* weights, uint32 numElements, uint32 sumOfWeights) {
    weights_ = weights;
    numElements_ = numElements;
    sumOfWeights_ = sumOfWeights;
}

template<class T>
DenseWeightVector<T>::~DenseWeightVector() {
    delete weights_;
}

template<class T>
uint32 DenseWeightVector<T>::getNumElements() {
    return numElements_;
}

template<class T>
bool DenseWeightVector<T>::hasZeroElements() {
    return true;
}

template<class T>
uint32 DenseWeightVector<T>::getValue(uint32 pos) {
    return (uint32) weights_[pos];
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

DenseIndexVector::DenseIndexVector(uint32* indices, uint32 numElements) {
    indices_ = indices;
    numElements_ = numElements;
}

DenseIndexVector::~DenseIndexVector() {
    delete[] indices_;
}

uint32 DenseIndexVector::getNumElements() {
    return numElements_;
}

bool DenseIndexVector::hasZeroElements() {
    return true;
}

uint32 DenseIndexVector::getIndex(uint32 pos) {
    return indices_[pos];
}

BaggingImpl::BaggingImpl(float32 sampleSize) {
    sampleSize_ = sampleSize;
}

std::unique_ptr<IWeightVector> BaggingImpl::subSample(uint32 numExamples, RNG* rng) {
    uint32 numSamples = (uint32) (sampleSize_ * numExamples);
    uint32* weights = new uint32[numExamples]{0};

    for (uint32 i = 0; i < numSamples; i++) {
        // Randomly select the index of an example...
        uint32 randomIndex = rng->random(0, numExamples);

        // Update weight at the selected index...
        weights[randomIndex] += 1;
    }

    return std::make_unique<DenseWeightVector<uint32>>(weights, numExamples, numSamples);
}

RandomInstanceSubsetSelectionImpl::RandomInstanceSubsetSelectionImpl(float32 sampleSize) {
    sampleSize_ = sampleSize;
}

std::unique_ptr<IWeightVector> RandomInstanceSubsetSelectionImpl::subSample(uint32 numExamples, RNG* rng) {
    uint32 numSamples = (uint32) (sampleSize_ * numExamples);
    return sampleWeightsWithoutReplacement(numExamples, numSamples, rng);
}

std::unique_ptr<IWeightVector> NoInstanceSubSamplingImpl::subSample(uint32 numExamples, RNG* rng) {
    return std::make_unique<EqualWeightVector>(numExamples);
}

RandomFeatureSubsetSelectionImpl::RandomFeatureSubsetSelectionImpl(float32 sampleSize) {
    sampleSize_ = sampleSize;
}

std::unique_ptr<IIndexVector> RandomFeatureSubsetSelectionImpl::subSample(uint32 numFeatures, RNG* rng) {
    uint32 numSamples;

    if (sampleSize_ > 0) {
            numSamples = (uint32) (sampleSize_ * numFeatures);
    } else {
            numSamples = (uint32) (log2(numFeatures - 1) + 1);
    }

    return sampleIndicesWithoutReplacement(numFeatures, numSamples, rng);
}

std::unique_ptr<IIndexVector> NoFeatureSubSamplingImpl::subSample(uint32 numFeatures, RNG* rng) {
    return std::make_unique<RangeIndexVector>(numFeatures);
}

RandomLabelSubsetSelectionImpl::RandomLabelSubsetSelectionImpl(uint32 numSamples) {
    numSamples_ = numSamples;
}

std::unique_ptr<IIndexVector> RandomLabelSubsetSelectionImpl::subSample(uint32 numLabels, RNG* rng) {
    return sampleIndicesWithoutReplacement(numLabels, numSamples_, rng);
}

std::unique_ptr<IIndexVector> NoLabelSubSamplingImpl::subSample(uint32 numLabels, RNG* rng) {
    return std::make_unique<RangeIndexVector>(numLabels);
}
