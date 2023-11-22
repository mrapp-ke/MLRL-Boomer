/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector.hpp"
#include "mlrl/common/indices/index_vector.hpp"

/**
 * Provides random access to a fixed number of indices stored in a C-contiguous array.
 */
class PartialIndexVector final
    : public ResizableVectorDecorator<IterableVectorDecorator<VectorDecorator<ResizableVector<uint32>>>>,
      public IIndexVector {
    public:

        /**
         * @param numElements   The number of elements in the vector
         * @param init          True, if all elements in the vector should be value-initialized, false otherwise
         */
        PartialIndexVector(uint32 numElements, bool init = false);

        uint32 getNumElements() const override;

        bool isPartial() const override;

        uint32 getIndex(uint32 pos) const override;

        std::unique_ptr<IRuleRefinement> createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                              uint32 featureIndex) const override;
};
