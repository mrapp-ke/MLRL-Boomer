/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../data/vector_sparse_array.hpp"
#include "../data/vector_dok_binary.hpp"


/**
 * An one-dimensional sparse vector that stores the values of training examples for a certain feature, as well as the
 * indices of examples with missing feature values.
 */
class FeatureVector final : public SparseArrayVector<float32> {

    private:

        BinaryDokVector missingIndices_;

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        FeatureVector(uint32 numElements);

        typedef BinaryDokVector::index_const_iterator missing_index_const_iterator;

        /**
         * Returns a `missing_index_const_iterator` to the beginning of the missing indices.
         *
         * @return A `missing_index_const_iterator` to the beginning
         */
        missing_index_const_iterator missing_indices_cbegin() const;

        /**
         * Returns a `missing_index_const_iterator` to the end of the missing indices.
         *
         * @return A `missing_index_const_iterator` to the end
         */
        missing_index_const_iterator missing_indices_cend() const;

        /**
         * Adds the index of an example with missing feature value.
         *
         * @param index The index to be added
         */
        void addMissingIndex(uint32 index);

        /**
         * Removes all indices of examples with missing feature values.
         */
        void clearMissingIndices();

};
