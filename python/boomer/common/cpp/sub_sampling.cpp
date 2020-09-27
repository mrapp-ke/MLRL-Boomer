#include "sub_sampling.h"
#include <unordered_set>


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

IWeightVector* NoInstanceSubSamplingImpl::subSample(uint32 numExamples, RNG* rng) {
    return new EqualWeightVector(numExamples);
}

IIndexVector* NoFeatureSubSamplingImpl::subSample(uint32 numFeatures, RNG* rng) {
    return new RangeIndexVector(numFeatures);
}


IIndexVector* NoLabelSubSamplingImpl::subSample(uint32 numLabels, RNG* rng) {
    return new RangeIndexVector(numLabels);
}
