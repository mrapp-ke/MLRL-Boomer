/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "feature_vector_decorator.hpp"
#include "feature_vector_nominal_allocated.hpp"
#include "mlrl/common/input/feature_vector_equal.hpp"

template<typename View, typename Decorator>
static inline std::unique_ptr<IFeatureVector> createFilteredNominalFeatureVectorDecorator(
  const View& view, std::unique_ptr<IFeatureVector>& existing, const CoverageMask& coverageMask) {
    std::unique_ptr<Decorator> filteredDecoratorPtr =
      createFilteredFeatureVectorDecorator<View, Decorator>(view, existing, coverageMask);

    // Filter the indices of examples not associated with the majority value...
    AllocatedNominalFeatureVector& filteredFeatureVector = filteredDecoratorPtr->getView().firstView;
    AllocatedNominalFeatureVector::index_iterator filteredIndexIterator = filteredFeatureVector.indices;
    AllocatedNominalFeatureVector::index_iterator filteredIndptrIterator = filteredFeatureVector.indptr;
    uint32 n = 0;

    for (uint32 i = 0; i < filteredFeatureVector.numValues; i++) {
        NominalFeatureVector::index_const_iterator indexIterator = view.getView().firstView.indices_cbegin(i);
        NominalFeatureVector::index_const_iterator indicesEnd = view.getView().firstView.indices_cend(i);
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
class AbstractNominalFeatureVectorDecorator : public AbstractFeatureVectorDecorator<AllocatedNominalFeatureVector> {
    public:

        /**
         * @param firstView   A reference to an object of template type `FeatureVector`
         * @param secondView  A reference to an object of type `AllocatedMissingFeatureVector`
         */
        AbstractNominalFeatureVectorDecorator(AllocatedNominalFeatureVector&& firstView,
                                              AllocatedMissingFeatureVector&& secondView)
            : AbstractFeatureVectorDecorator<AllocatedNominalFeatureVector>(std::move(firstView),
                                                                            std::move(secondView)) {}

        /**
         * @param A reference to an object of type `AbstractNominalFeatureVectorDecorator` that should be copied
         */
        AbstractNominalFeatureVectorDecorator(const AbstractNominalFeatureVectorDecorator& other)
            : AbstractNominalFeatureVectorDecorator(
              AllocatedNominalFeatureVector(other.view.firstView.numValues,
                                            other.view.firstView.indptr[other.view.firstView.numValues],
                                            other.view.firstView.majorityValue),
              AllocatedMissingFeatureVector()) {}

        virtual ~AbstractNominalFeatureVectorDecorator() override {}
};
