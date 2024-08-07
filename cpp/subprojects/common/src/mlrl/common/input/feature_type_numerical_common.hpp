/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_vector_decorator_numerical.hpp"
#include "mlrl/common/data/view_matrix_csc.hpp"
#include "mlrl/common/data/view_matrix_fortran_contiguous.hpp"
#include "mlrl/common/iterator/iterator_index.hpp"

#include <algorithm>
#include <memory>
#include <utility>

template<typename IndexIterator, typename ValueIterator>
static inline std::unique_ptr<NumericalFeatureVectorDecorator> createNumericalFeatureVector(IndexIterator indexIterator,
                                                                                            ValueIterator valueIterator,
                                                                                            uint32 numElements) {
    AllocatedNumericalFeatureVector numericalFeatureVector(numElements);
    AllocatedMissingFeatureVector missingFeatureVector;
    uint32 n = 0;

    for (uint32 i = 0; i < numElements; i++) {
        uint32 index = indexIterator[i];
        float32 value = valueIterator[i];

        if (std::isnan(value)) {
            missingFeatureVector.set(index, true);
        } else {
            IndexedValue<float32>& entry = numericalFeatureVector[n];
            entry.index = index;
            entry.value = value;
            n++;
        }
    }

    numericalFeatureVector.resize(n, true);
    std::sort(numericalFeatureVector.begin(), numericalFeatureVector.end(), IndexedValue<float32>::CompareValue());
    return std::make_unique<NumericalFeatureVectorDecorator>(std::move(numericalFeatureVector),
                                                             std::move(missingFeatureVector));
}

static inline std::unique_ptr<NumericalFeatureVectorDecorator> createNumericalFeatureVector(
  uint32 featureIndex, const FortranContiguousView<const float32>& featureMatrix) {
    FortranContiguousView<const float32>::value_const_iterator valueIterator =
      featureMatrix.values_cbegin(featureIndex);
    uint32 numRows = featureMatrix.numRows;
    return createNumericalFeatureVector(IndexIterator(), valueIterator, numRows);
}

static inline std::unique_ptr<NumericalFeatureVectorDecorator> createNumericalFeatureVector(
  uint32 featureIndex, const CscView<const float32>& featureMatrix) {
    CscView<const float32>::index_const_iterator indexIterator = featureMatrix.indices_cbegin(featureIndex);
    CscView<const float32>::index_const_iterator indicesEnd = featureMatrix.indices_cend(featureIndex);
    CscView<const float32>::value_const_iterator valueIterator = featureMatrix.values_cbegin(featureIndex);
    uint32 numIndices = indicesEnd - indexIterator;
    return createNumericalFeatureVector(indexIterator, valueIterator, numIndices);
}
