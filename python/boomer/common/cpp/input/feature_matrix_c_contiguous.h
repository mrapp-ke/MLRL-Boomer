/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../data/types.h"


/**
 * Implements row-wise access to the feature values of individual training examples that are stored in a pre-allocated
 * C-contiguous array.
 */
class CContiguousFeatureMatrix final {

    private:

        uint32 numExamples_;

        uint32 numFeatures_;

        const float32* x_;

    public:

        /**
         * @param numExamples   The number of examples
         * @param numFeatures   The number of features
         * @param x             A pointer to a C-contiguous array of type `float32`, shape `(numExamples, numFeatures)`,
         *                      representing the feature values of the training examples
         */
        CContiguousFeatureMatrix(uint32 numExamples, uint32 numFeatures, const float32* x);

        typedef const float32* const_iterator;

        /**
         * Returns a `const_iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the beginning of the given row
         */
        const_iterator row_cbegin(uint32 row) const;

        /**
         * Returns a `const_iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the end of the given row
         */
        const_iterator row_cend(uint32 row) const;

        /**
         * Returns the number of available examples.
         *
         * @return The number of examples
         */
        uint32 getNumExamples() const;

        /**
         * Returns the number of available features.
         *
         * @return The number of features
         */
        uint32 getNumFeatures() const;

};
