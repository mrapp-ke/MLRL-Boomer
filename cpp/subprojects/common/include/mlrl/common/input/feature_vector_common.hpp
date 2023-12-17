/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_dok_binary.hpp"
#include "mlrl/common/input/feature_vector.hpp"

#include <memory>

/**
 * An abstract base class for all feature vectors that store the values of training examples for a certain feature. It
 * allows to keep track of the indices of examples with missing feature values.
 */
class AbstractFeatureVector : public IFeatureVector {
    private:

        BinaryDokVector missingIndices_;

    public:

        virtual ~AbstractFeatureVector() override {}

        /**
         * An iterator that provides read-only access to the indices of examples with missing feature values.
         */
        typedef BinaryDokVector::index_const_iterator missing_index_const_iterator;

        /**
         * Returns a `missing_index_const_iterator` to the beginning of the indices of examples with missing feature
         * values.
         *
         * @return A `missing_index_const_iterator` to the beginning
         */
        missing_index_const_iterator missing_indices_cbegin() const;

        /**
         * Returns a `missing_index_const_iterator` to the end of the indices of examples with missing feature values.
         *
         * @return A `missing_index_const_iterator` to the end
         */
        missing_index_const_iterator missing_indices_cend() const;

        /**
         * Sets whether the example at a specific index is missing a feature value or not.
         *
         * @param index     The index of the example
         * @param missing   True, if the example at the given index is missing a feature value, false otherwise
         */
        void setMissing(uint32 index, bool missing);

        /**
         * Returns whether the example at a specific index is missing a feature value or not.
         *
         * @param index The index of the example
         * @return      True, if the example at the given index is missing a feature value, false otherwise
         */
        bool isMissing(uint32 index) const;
};
