/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../indices/index_vector_partial.h"
#include <unordered_set>


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
