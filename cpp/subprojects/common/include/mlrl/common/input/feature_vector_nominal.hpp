/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_vector_common.hpp"

/**
 * A feature vector that stores the indices of the examples that are associated with each value, except for the majority
 * value, i.e., the most frequent value, of a nominal feature.
 */
class NominalFeatureVector final : public AbstractFeatureVector {
    private:

        int32* values_;

        uint32* indices_;

        uint32* indptr_;

        const uint32 numValues_;

        const int32 majorityValue_;

    public:

        /**
         * @param numValues     The number of distinct values of the nominal feature, excluding the majority value
         * @param numExamples   The total number of examples, excluding those associated with the majority value
         * @param majorityValue The majority value, i.e., the most frequent value, of the nominal feature
         */
        NominalFeatureVector(uint32 numValues, uint32 numExamples, int32 majorityValue);

        ~NominalFeatureVector() override;

        /**
         * An iterator that provides access to the values of the nominal feature and allows to modify them.
         */
        typedef int32* value_iterator;

        /**
         * An iterator that provides read-only access to the values of the nominal feature.
         */
        typedef const int32* value_const_iterator;

        /**
         * An iterator that provides access to the indices of the examples that are associated with each value of the
         * nominal feature, except for the majority value, and allows to modify them.
         */
        typedef uint32* index_iterator;

        /**
         * An iterator that provides read-only access to the indices of the examples that are associated with each value
         * of the nominal feature, except for the majority value.
         */
        typedef const uint32* index_const_iterator;

        /**
         * An iterator that provides access to the indices that specify the first element in the array of example
         * indices that corresponds to each value of the nominal feature.
         */
        typedef uint32* indptr_iterator;

        /**
         * Returns a `value_iterator` to the beginning of the values of the nominal feature.
         *
         * @return A `value_iterator` to the beginning
         */
        value_iterator values_begin();

        /**
         * Returns a `value_iterator` to the end of the values of the nominal feature.
         *
         * @return A `value_iterator` to the end
         */
        value_iterator values_end();

        /**
         * Returns a `value_const_iterator` to the beginning of the values of the nominal feature.
         *
         * @return A `value_const_iterator` to the beginning
         */
        value_const_iterator values_cbegin() const;

        /**
         * Returns a `value_const_iterator` to the end of the value of the nominal feature.
         *
         * @return A `value_const_iterator` to the end
         */
        value_const_iterator values_cend() const;

        /**
         * Returns an `index_iterator` to the beginning of the indices of the examples that are associated with a
         * specific value of the nominal feature.
         *
         * @param index The index of the value
         * @return      An `index_iterator` to the beginning
         */
        index_iterator indices_begin(uint32 index);

        /**
         * Returns an `index_iterator` to the end of the indices of the examples that are associated with a specific
         * value of the nominal feature.
         *
         * @param index The index of the value
         * @return      An `index_iterator` to the end
         */
        index_iterator indices_end(uint32 index);

        /**
         * Returns an `index_const_iterator` to the beginning of the indices of the examples that are associated with a
         * specific value of the nominal feature.
         *
         * @param index The index of the value
         * @return      An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin(uint32 index) const;

        /**
         * Returns an `index_const_iterator` to the end of the indices of the examples that are associated with a
         * specific value of the nominal feature.
         *
         * @param index The index of the value
         * @return      An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend(uint32 index) const;

        /**
         * Returns an `indptr_iterator` to the beginning of the indices that specify the first element in the array of
         * example indices that corresponds to each value of the nominal feature, except for the majority value.
         *
         * @return An `indptr_iterator` to the beginning
         */
        indptr_iterator indptr_begin();

        /**
         * Returns an `indptr_iterator` to the end of the indices that specify the first element in the array of example
         * indices that corresponds to each value of the nominal feature, except for the majority value.
         *
         * @return An `indptr_iterator` to the end
         */
        indptr_iterator indptr_end();

        /**
         * Returns the majority value, i.e., the least frequent value, of the nominal feature.
         *
         * @return The majority value
         */
        int32 getMajorityValue() const;

        uint32 getNumElements() const override;
};
