/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_vector_nominal_allocated.hpp"
#include "mlrl/common/input/feature_vector_equal.hpp"

#include <memory>

template<typename View, typename Decorator>
static inline std::unique_ptr<IFeatureVector> createFilteredNominalFeatureVectorDecorator(
  const View& view, std::unique_ptr<IFeatureVector>& existing, const CoverageMask& coverageMask) {
    std::unique_ptr<Decorator> filteredDecoratorPtr =
      createFilteredFeatureVectorDecorator<View, Decorator>(view, existing, coverageMask);

    // Filter the indices of examples not associated with the majority value...
    const NominalFeatureVector& featureVector = view.getView().firstView;
    AllocatedNominalFeatureVector& filteredFeatureVector = filteredDecoratorPtr->getView().firstView;
    AllocatedNominalFeatureVector::index_iterator filteredIndexIterator = filteredFeatureVector.indices;
    AllocatedNominalFeatureVector::index_iterator filteredIndptrIterator = filteredFeatureVector.indptr;
    AllocatedNominalFeatureVector::value_iterator filteredValueIterator = filteredFeatureVector.values;
    uint32 numFilteredValues = 0;
    uint32 numFilteredIndices = 0;

    for (uint32 i = 0; i < featureVector.numBins; i++) {
        NominalFeatureVector::index_const_iterator indexIterator = featureVector.indices_cbegin(i);
        NominalFeatureVector::index_const_iterator indicesEnd = featureVector.indices_cend(i);
        uint32 numIndices = indicesEnd - indexIterator;
        uint32 indptr = numFilteredIndices;

        for (uint32 j = 0; j < numIndices; j++) {
            uint32 index = indexIterator[j];

            if (coverageMask[index]) {
                filteredIndexIterator[numFilteredIndices] = index;
                numFilteredIndices++;
            }
        }

        if (numFilteredIndices > indptr) {
            filteredIndptrIterator[numFilteredValues] = indptr;
            filteredValueIterator[numFilteredValues] = featureVector.values[i];
            numFilteredValues++;
        }
    }

    if (numFilteredIndices > 0) {
        filteredFeatureVector.resize(numFilteredValues, numFilteredIndices);
        return filteredDecoratorPtr;
    }

    return std::make_unique<EqualFeatureVector>();
}
