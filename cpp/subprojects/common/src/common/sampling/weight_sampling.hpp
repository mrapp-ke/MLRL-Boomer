/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/sampling/weight_vector_dense.hpp"
#include "common/data/arrays.hpp"
#include <unordered_set>


/**
 * Randomly selects `numSamples` out of `numTotal` elements and sets their weights to 1, while the remaining weights are
 * set to 0, by using a set to keep track of the elements that have already been selected. This method is suitable if
 * `numSamples` is much smaller than `numTotal`.
 *
 * @tparam Iterator     The type of the iterator that provides random access to the indices of the available elements to
 *                      sample from
 * @param weightVector  A reference to an object of type `DenseWeightVector` the weights should be written to
 * @param iterator      An iterator that provides random access to the indices of the available elements to sample from
 * @param numTotal      The total number of available elements to sample from
 * @param numSamples    The number of weights to be set to 1
 * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be used
 */
template<class Iterator>
static inline void sampleWeightsWithoutReplacementViaTrackingSelection(DenseWeightVector<uint8>& weightVector,
                                                                       Iterator iterator, uint32 numTotal,
                                                                       uint32 numSamples, RNG& rng) {
    typename DenseWeightVector<uint8>::iterator sampleIterator = weightVector.begin();
    setArrayToZeros(sampleIterator, weightVector.getNumElements());
    std::unordered_set<uint32> selectedIndices;

    for (uint32 i = 0; i < numSamples; i++) {
        bool shouldContinue = true;
        uint32 sampledIndex;

        while (shouldContinue) {
            uint32 randomIndex = rng.random(0, numTotal);
            sampledIndex = iterator[randomIndex];
            shouldContinue = !selectedIndices.insert(sampledIndex).second;
        }

        sampleIterator[sampledIndex] = 1;
    }

    weightVector.setNumNonZeroWeights(numSamples);
}

/**
 * Randomly selects `numSamples` out of `numTotal` elements and sets their weights to 1, while the remaining weights are
 * set to 0, by using a pool, i.e., an array, to keep track of the elements that have not been selected yet.
 *
 * @tparam Iterator     The type of the iterator that provides random access to the indices of the available elements to
 *                      sample from
 * @param weightVector  A reference to an object of type `DenseWeightVector` the weights should be written to
 * @param iterator      An iterator that provides random access to the indices of the available elements to sample from
 * @param numTotal      The total number of available elements to sample from
 * @param numSamples    The number of weights to be set to 1
 * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be used
 */
template<class Iterator>
static inline void sampleWeightsWithoutReplacementViaPool(DenseWeightVector<uint8>& weightVector, Iterator iterator,
                                                          uint32 numTotal, uint32 numSamples, RNG& rng) {
    typename DenseWeightVector<uint8>::iterator sampleIterator = weightVector.begin();
    setArrayToZeros(sampleIterator, weightVector.getNumElements());
    uint32 pool[numTotal];

    // Initialize pool...
    for (uint32 i = 0; i < numTotal; i++) {
        pool[i] = iterator[i];
    }

    for (uint32 i = 0; i < numSamples; i++) {
        // Randomly select an index that has not been drawn yet...
        uint32 randomIndex = rng.random(0, numTotal - i);
        uint32 sampledIndex = pool[randomIndex];

        // Set weight at the selected index to 1...
        sampleIterator[sampledIndex] = 1;

        // Move the index at the border to the position of the recently drawn index...
        pool[randomIndex] = pool[numTotal - i - 1];
    }

    weightVector.setNumNonZeroWeights(numSamples);
}

/**
 * Randomly selects `numSamples` out of `numTotal` elements and sets their weights to 1, while the remaining weights are
 * set to 0. The method that is used internally is chosen automatically, depending on the ratio `numSamples / numTotal`.
 *
 * @tparam Iterator     The type of the iterator that provides random access to the indices of the available elements to
 *                      sample from
 * @param weightVector  A reference to an object of type `DenseWeightVector` the weights should be written to
 * @param iterator      An iterator that provides random access to the indices of the available elements to sample from
 * @param numTotal      The total number of available elements to sample from
 * @param numSamples    The number of weights to be set to 1
 * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be used
 *
 */
template<class Iterator>
static inline void sampleWeightsWithoutReplacement(DenseWeightVector<uint8>& weightVector, Iterator iterator,
                                                   uint32 numTotal, uint32 numSamples, RNG& rng) {
    float64 ratio = numTotal > 0 ? ((float64) numSamples) / ((float64) numTotal) : 1;

    if (ratio < 0.06) {
        // For very small ratios use tracking selection
        sampleWeightsWithoutReplacementViaTrackingSelection(weightVector, iterator, numTotal, numSamples, rng);
    } else {
        // Otherwise, use a pool as the default method
        sampleWeightsWithoutReplacementViaPool(weightVector, iterator, numTotal, numSamples, rng);
    }
}
