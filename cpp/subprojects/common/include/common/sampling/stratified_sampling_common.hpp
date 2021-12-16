/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/label_matrix_c_contiguous.hpp"
#include "common/input/label_matrix_csr.hpp"
#include "common/sampling/random.hpp"
#include <unordered_map>


static inline void updateNumExamplesPerLabel(const CContiguousLabelMatrix& labelMatrix, uint32 exampleIndex,
                                             uint32* numExamplesPerLabel,
                                             std::unordered_map<uint32, uint32>& affectedLabelIndices) {
    CContiguousLabelMatrix::value_const_iterator labelIterator = labelMatrix.row_values_cbegin(exampleIndex);
    uint32 numLabels = labelMatrix.getNumCols();

    for (uint32 i = 0; i < numLabels; i++) {
        if (labelIterator[i]) {
            uint32 numRemaining = numExamplesPerLabel[i];
            numExamplesPerLabel[i] = numRemaining - 1;
            affectedLabelIndices.emplace(i, numRemaining);
        }
    }
}

static inline void updateNumExamplesPerLabel(const CsrLabelMatrix& labelMatrix, uint32 exampleIndex,
                                             uint32* numExamplesPerLabel,
                                             std::unordered_map<uint32, uint32>& affectedLabelIndices) {
    CsrLabelMatrix::index_const_iterator indexIterator = labelMatrix.row_indices_cbegin(exampleIndex);
    uint32 numLabels = labelMatrix.row_indices_cend(exampleIndex) - indexIterator;

    for (uint32 i = 0; i < numLabels; i++) {
        uint32 labelIndex = indexIterator[i];
        uint32 numRemaining = numExamplesPerLabel[labelIndex];
        numExamplesPerLabel[labelIndex] = numRemaining - 1;
        affectedLabelIndices.emplace(labelIndex, numRemaining);
    }
}

static inline bool tiebreak(uint32 numDesiredSamples, uint32 numDesiredOutOfSamples, RNG& rng) {
    if (numDesiredSamples > numDesiredOutOfSamples) {
        return true;
    } else if (numDesiredSamples < numDesiredOutOfSamples) {
        return false;
    } else {
        return rng.random(0, 2) != 0;
    }
}
