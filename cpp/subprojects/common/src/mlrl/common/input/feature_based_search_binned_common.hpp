/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/statistics/statistics_subset.hpp"

/**
 * Adds all examples corresponding to a single bin in a given feature vector to a given `IStatisticsSubset`, if they
 * have non-zero weights.
 *
 * @tparam FeatureVector    The type of the feature vector
 * @param statisticsSubset  A reference to an object of type `IStatisticsSubset`
 * @param featureVector     A reference to an object of template type `FeatureVector`´that stores the indices of the
 *                          examples that correspond to individual bins
 * @param index             The index of the bin
 * @return                  The number of examples with non-zero weights
 */
template<typename FeatureVector>
static inline uint32 addAllToSubset(IStatisticsSubset& statisticsSubset, const FeatureVector& featureVector,
                                    uint32 index) {
    typename FeatureVector::index_const_iterator indexIterator = featureVector.indices_cbegin(index);
    typename FeatureVector::index_const_iterator indicesEnd = featureVector.indices_cend(index);
    uint32 numIndices = indicesEnd - indexIterator;
    uint32 numUncovered = 0;

    for (uint32 i = 0; i < numIndices; i++) {
        uint32 exampleIndex = indexIterator[i];

        // Do only consider examples with non-zero weights...
        if (statisticsSubset.hasNonZeroWeight(exampleIndex)) {
            statisticsSubset.addToSubset(exampleIndex);
        } else {
            numUncovered++;
        }
    }

    return numIndices - numUncovered;
}
