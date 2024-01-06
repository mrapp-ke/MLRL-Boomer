/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_vector_decorator.hpp"
#include "feature_vector_nominal_allocated.hpp"
#include "mlrl/common/input/feature_vector_equal.hpp"

template<typename Decorator>
static inline std::unique_ptr<IFeatureVector> createFilteredNominalFeatureVectorDecorator(
  const Decorator& decorator, std::unique_ptr<IFeatureVector>& existing, const CoverageMask& coverageMask) {
    std::unique_ptr<Decorator> filteredDecoratorPtr;
    Decorator* existingDecorator = dynamic_cast<Decorator*>(existing.get());

    if (existingDecorator) {
        // Reuse the existing feature vector...
        existing.release();
        filteredDecoratorPtr = std::unique_ptr<Decorator>(existingDecorator);

        // Filter the indices of examples with missing feature values...
        MissingFeatureVector& missingFeatureVector = filteredDecoratorPtr->getView().secondView;

        for (auto it = missingFeatureVector.indices_cbegin(); it != missingFeatureVector.indices_cend();) {
            uint32 index = *it;
            it++;  // Iterator must be incremented before call to `MissingFeatureVector::set` invalidates it

            if (!coverageMask.isCovered(index)) {
                missingFeatureVector.set(index, false);
            }
        }
    } else {
        // Create a new feature vector...
        filteredDecoratorPtr = std::make_unique<Decorator>(decorator);

        // Add the indices of examples with missing feature values...
        MissingFeatureVector& missingFeatureVector = filteredDecoratorPtr->getView().secondView;

        for (auto it = decorator.getView().secondView.indices_cbegin();
             it != decorator.getView().secondView.indices_cend(); it++) {
            uint32 index = *it;

            if (coverageMask.isCovered(index)) {
                missingFeatureVector.set(index, true);
            }
        }
    }

    // Filter the indices of examples not associated with the majority value...
    AllocatedNominalFeatureVector& filteredFeatureVector = filteredDecoratorPtr->getView().firstView;
    AllocatedNominalFeatureVector::index_iterator filteredIndexIterator = filteredFeatureVector.indices;
    AllocatedNominalFeatureVector::index_iterator filteredIndptrIterator = filteredFeatureVector.indptr;
    uint32 n = 0;

    for (uint32 i = 0; i < filteredFeatureVector.numValues; i++) {
        NominalFeatureVector::index_const_iterator indexIterator = decorator.getView().firstView.indices_cbegin(i);
        NominalFeatureVector::index_const_iterator indicesEnd = decorator.getView().firstView.indices_cend(i);
        uint32 numIndices = indicesEnd - indexIterator;
        filteredIndptrIterator[i] = n;

        for (uint32 j = 0; j < numIndices; j++) {
            uint32 index = indexIterator[j];

            if (coverageMask.isCovered(index)) {
                filteredIndexIterator[n] = index;
                n++;
            }
        }
    }

    if (n > 0) {
        filteredFeatureVector.resize(n);
        return filteredDecoratorPtr;
    }

    return std::make_unique<EqualFeatureVector>();
}

/**
 * An abstract base class for all decorators that provide access to the values and indices of training examples stored
 * in an `AllocatedNominalFeatureVector`.
 */
typedef AbstractFeatureVectorDecorator<AllocatedNominalFeatureVector> AbstractNominalFeatureVectorDecorator;
