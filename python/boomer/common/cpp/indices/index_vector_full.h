/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "index_vector.h"


/**
 * Provides random access to all indices within a continuous range [0, numIndices).
 */
class FullIndexVector : public IIndexVector {

    private:

        uint32 numElements_;

    public:

        /**
         * Allows to iterate the indices of a `FullIndexVector`.
         */
        class Iterator {

            private:

                uint32 index_;

            public:

                Iterator(uint32 index);

                uint32 operator[](uint32 index) const;

                uint32 operator*() const;

                Iterator& operator++(int n);

                bool operator!=(const Iterator& rhs) const;

        };

        /**
         * @param numElements The number of indices, the vector provides access to
         */
        FullIndexVector(uint32 numElements);

        typedef Iterator const_iterator;

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

        std::unique_ptr<IStatisticsSubset> createSubset(const IHistogram& histogram) const override;

        std::unique_ptr<IRuleRefinement> createRuleRefinement(IThresholdsSubset& thresholdsSubset,
                                                              uint32 featureIndex) const override;

};
