/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/indices/index_vector_partial.hpp"
#include <unordered_set>


/**
 * Randomly selects `numSamples` out of `numTotal` indices without replacement by using a set to keep track of the
 * indices that have already been selected. This method is suitable if `numSamples` is much smaller than `numTotal`
 *
 * @tparam Iterator     The type of the iterator that provides random access to the available indices to sample from
 * @param indexVector   A reference to an object of type `PartialIndexVector`, the sampled indices should be written to
 * @param iterator      An iterator that provides random access to the available indices to sample from
 * @param numTotal      The total number of available indices to sample from
 * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be used
 */
template<class Iterator>
static inline void sampleIndicesWithoutReplacementViaTrackingSelection(PartialIndexVector& indexVector,
                                                                       Iterator iterator, uint32 numTotal, RNG& rng) {
    uint32 numSamples = indexVector.getNumElements();
    PartialIndexVector::iterator sampleIterator = indexVector.begin();
    std::unordered_set<uint32> selectedIndices;

    for (uint32 i = 0; i < numSamples; i++) {
        bool shouldContinue = true;
        uint32 sampledIndex;

        while (shouldContinue) {
            uint32 randomIndex = rng.random(0, numTotal);
            sampledIndex = iterator[randomIndex];
            shouldContinue = !selectedIndices.insert(sampledIndex).second;
        }

        sampleIterator[i] = sampledIndex;
    }
}

/**
 * Randomly selects `numSamples` out of `numTotal` indices without replacement using a reservoir sampling algorithm.
 * This method is suitable if `numSamples` is almost as large as `numTotal`.
 *
 * @tparam Iterator     The type of the iterator that provides random access to the available indices to sample from
 * @param indexVector   A reference to an object of type `PartialIndexVector`, the sampled indices should be written to
 * @param iterator      An iterator that provides random access to the available indices to sample from
 * @param numTotal      The total number of available indices to sample from
 * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be used
 */
template<class Iterator>
static inline void sampleIndicesWithoutReplacementViaReservoirSampling(PartialIndexVector& indexVector,
                                                                       Iterator iterator, uint32 numTotal, RNG& rng) {
    uint32 numSamples = indexVector.getNumElements();
    PartialIndexVector::iterator sampleIterator = indexVector.begin();

    for (uint32 i = 0; i < numSamples; i++) {
        sampleIterator[i] = iterator[i];
    }

    for (uint32 i = numSamples; i < numTotal; i++) {
        uint32 randomIndex = rng.random(0, i + 1);

        if (randomIndex < numSamples) {
            sampleIterator[randomIndex] = iterator[i];
        }
    }
}

/**
 * Computes a random permutation of the indices that are contained by two mutually exclusive sets using the Fisher-Yates
 * shuffle.
 *
 * @tparam FirstIterator    The type of the iterator that provides random access to the indices that are contained by
 *                          the first set
 * @tparam SecondIterator   The type of the iterator that provides random access to the indices that are contained by
 *                          the second set
 * @param firstIterator     The iterator that provides random access to the indices that are contained by the first set
 * @param secondIterator    The iterator that provides random access to the indices that are contained by the second set
 * @param numFirst          The number of indices that are contained by the first set
 * @param numTotal          The total number of indices to sample from
 * @param numPermutations   The maximum number of permutations to be performed. Must be in [1, numTotal)
 * @param rng               A reference to an object of type `RNG`, implementing the random number generator to be used
 */
template<class FirstIterator, class SecondIterator>
static inline void randomPermutation(FirstIterator firstIterator, SecondIterator secondIterator, uint32 numFirst,
                                     uint32 numTotal, uint32 numPermutations, RNG& rng) {
    for (uint32 i = 0; i < numPermutations; i++) {
        // Swap elements at index i and at a randomly selected index...
        uint32 randomIndex = rng.random(i, numTotal);
        uint32 tmp1 = i < numFirst ? firstIterator[i] : secondIterator[i - numFirst];
        uint32 tmp2;

        if (randomIndex < numFirst) {
            tmp2 = firstIterator[randomIndex];
            firstIterator[randomIndex] = tmp1;
        } else {
            tmp2 = secondIterator[randomIndex - numFirst];
            secondIterator[randomIndex - numFirst] = tmp1;
        }

        if (i < numFirst) {
            firstIterator[i] = tmp2;
        } else {
            secondIterator[i - numFirst] = tmp2;
        }
    }
}

/**
 * Randomly selects `numSamples` out of `numTotal` indices without replacement by first generating a random permutation
 * of the available indices and then returning the first `numSamples` indices.
 *
 * @tparam Iterator     The type of the iterator that provides random access to the available indices to sample from
 * @param indexVector   A reference to an object of type `PartialIndexVector`, the sampled indices should be written to
 * @param iterator      An iterator that provides random access to the available indices to sample from
 * @param numTotal      The total number of available indices to sample from
 * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be used
 */
template<class Iterator>
static inline void sampleIndicesWithoutReplacementViaRandomPermutation(PartialIndexVector& indexVector,
                                                                       Iterator iterator, uint32 numTotal, RNG& rng) {
    uint32 numSamples = indexVector.getNumElements();
    PartialIndexVector::iterator sampleIterator = indexVector.begin();
    uint32 unusedIndices[numTotal - numSamples];

    for (uint32 i = 0; i < numSamples; i++) {
        sampleIterator[i] = iterator[i];
    }

    for (uint32 i = numSamples; i < numTotal; i++) {
        unusedIndices[i - numSamples] = iterator[i];
    }

    randomPermutation<PartialIndexVector::iterator, uint32*>(sampleIterator, &unusedIndices[0], numSamples, numTotal,
                                                             numSamples, rng);
}

/**
 * Randomly selects `numSamples` out of `numTotal` indices without replacement. The method that is used internally is
 * chosen automatically, depending on the ratio `numSamples / numTotal`.
 *
 * @tparam Iterator     The type of the iterator that provides random access to the available indices to sample from
 * @param indexVector   A reference to an object of type `PartialIndexVector`, the sampled indices should be written to
 * @param iterator      An iterator that provides random access to the available indices to sample from
 * @param numTotal      The total number of available indices to sample from
 * @param rng           A reference to an object of type `RNG`, implementing the random number generator to be used
 */
template<class Iterator>
static inline void sampleIndicesWithoutReplacement(PartialIndexVector& indexVector, Iterator iterator, uint32 numTotal,
                                                   RNG& rng) {
    float64 ratio = numTotal > 0 ? ((float64) indexVector.getNumElements()) / ((float64) numTotal) : 1;

    // The thresholds for choosing a suitable method are based on empirical experiments
    if (ratio < 0.06) {
        // For very small ratios use tracking selection
        sampleIndicesWithoutReplacementViaTrackingSelection(indexVector, iterator, numTotal, rng);
    } else if (ratio > 0.5) {
        // For large ratios use reservoir sampling
        sampleIndicesWithoutReplacementViaReservoirSampling(indexVector, iterator, numTotal, rng);
    } else {
        // Otherwise, use random permutation as the default method
        sampleIndicesWithoutReplacementViaRandomPermutation(indexVector, iterator, numTotal, rng);
    }
}
