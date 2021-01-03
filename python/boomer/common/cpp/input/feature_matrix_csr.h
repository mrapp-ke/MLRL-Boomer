/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../data/types.h"


/**
 * Implements row-wise read-only access to the feature values of individual training examples that are stored in a
 * pre-allocated sparse matrix in the compressed sparse row (CSR) format.
 */
class CsrFeatureMatrix final {

    private:

        uint32 numExamples_;

        uint32 numFeatures_;

        const float32* xData_;

        const uint32* xRowIndices_;

        const uint32* xColIndices_;

    public:

        /**
         * @param numExamples   The number of examples
         * @param numFeatures   The number of features
         * @param xData         A pointer to an array of type `float32`, shape `(num_non_zero_feature_values)`,
         *                      representing the non-zero feature values of the training examples
         * @param xRowIndices   A pointer to an array of type `uint32`, shape `(num_examples + 1)`, representing the
         *                      indices of the first element in `xData` and `xColIndices` that corresponds to a certain
         *                      example. The index at the last position is equal to `num_non_zero_feature_values`
         * @param xColIndices   A pointer to an array of type `uint32`, shape `(num_non_zero_feature_values)`,
         *                      representing the column-indices of the features, the values in `xData` correspond to
         */
        CsrFeatureMatrix(uint32 numExamples, uint32 numFeatures, const float32* xData, const uint32* xRowIndices,
                         const uint32* xColIndices);

        typedef const float32* value_const_iterator;

        typedef const uint32* index_const_iterator;

        /**
         * Returns a `value_const_iterator` to the beginning of the values at a specific row.
         *
         * @param row   The row
         * @return      A `value_const_iterator` to the beginning of the values
         */
        value_const_iterator row_values_cbegin(uint32 row) const;

        /**
         * Returns a `value_const_iterator` to the end of the values at a specific row.
         *
         * @param row   The row
         * @return      A `value_const_iterator` to the end of the values
         */
        value_const_iterator row_values_cend(uint32 row) const;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices at a specific row.
         *
         * @param row   The row
         * @return      An `index_const_iterator` to the beginning of the indices
         */
        index_const_iterator row_indices_cbegin(uint32 row) const;

        /**
         * Returns an `index_const_iterator` to the end of the indices at a specific row.
         *
         * @param row   The row
         * @return      An `index_const_iterator` to the end of the indices
         */
        index_const_iterator row_indices_cend(uint32 row) const;

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
