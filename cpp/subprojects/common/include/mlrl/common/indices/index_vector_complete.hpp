/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/indices/index_vector.hpp"
#include "mlrl/common/iterator/iterator_index.hpp"

#include <memory>

/**
 * Provides random access to all indices within a continuous range [0, numIndices).
 */
class CompleteIndexVector final : public IIndexVector {
    private:

        uint32 numElements_;

    public:

        /**
         * @param numElements The number of indices, the vector provides access to
         */
        CompleteIndexVector(uint32 numElements);

        /**
         * An iterator that provides read-only access to the indices in the vector.
         */
        typedef IndexIterator const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of the indices.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the indices.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const;

        /**
         * Sets the number of indices.
         *
         * @param numElements   The number of indices to be set
         * @param freeMemory    True, if unused memory should be freed, if possible, false otherwise
         */
        void setNumElements(uint32 numElements, bool freeMemory);

        uint32 getNumElements() const override;

        bool isPartial() const override;

        uint32 getIndex(uint32 pos) const override;

        std::unique_ptr<IRuleRefinement> createRuleRefinement(IFeatureSubspace& featureSubspace, uint32 featureIndex,
                                                              const IWeightedStatistics& statistics,
                                                              const IFeatureVector& featureVector) const override;

        void visit(PartialIndexVectorVisitor partialIndexVectorVisitor,
                   CompleteIndexVectorVisitor completeIndexVectorVisitor) const override;
};
