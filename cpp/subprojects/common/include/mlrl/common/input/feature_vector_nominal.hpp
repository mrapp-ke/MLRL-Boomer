/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view.hpp"

/**
 * A feature vector that stores the indices of the examples that are associated with each value, except for the majority
 * value, i.e., the most frequent value, of a nominal feature.
 */
class MLRLCOMMON_API NominalFeatureVector {
    public:

        /**
         * A pointer to an array that stores all nominal values.
         */
        int32* values;

        /**
         * A pointer to an array that stores the indices of all examples not associated with the majority value.
         */
        uint32* indices;

        /**
         * A pointer to an array that stores the indices of the first element in `indices` that corresponds to a certain
         * value in `values`.
         */
        uint32* indptr;

        /**
         * The number of nominal values.
         */
        uint32 numValues;

        /**
         * The majority value, i.e., the most frequent value, of the nominal feature.
         */
        int32 majorityValue;

        /**
         * @param values        A pointer to an array of type `int32`, shape `(numValues)` that stores all nominal
         *                      values
         * @param indices       A pointer to an array of type `uint32`, shape `(numIndices)` that stores the indices of
         *                      all examples not associated with the majority value
         * @param indptr        A pointer to an array that stores the indices of the first element in `indices` that
         *                      corresponds to a certain value in `values`
         * @param numValues     The number of elements in the array `values`
         * @param numIndices    The number of elements in the array `indices`
         * @param majorityValue The majority value, i.e., the most frequent value, of the nominal feature
         */
        NominalFeatureVector(int32* values, uint32* indices, uint32* indptr, uint32 numValues, uint32 numIndices,
                             int32 majorityValue);

        /**
         * @param other A reference to an object of type `NominalFeatureVector` that should be copied
         */
        NominalFeatureVector(const NominalFeatureVector& other);

        /**
         * @param other A reference to an object of type `NominalFeatureVector` that should be moved
         */
        NominalFeatureVector(NominalFeatureVector&& other);

        virtual ~NominalFeatureVector() {}

        /**
         * The type of the indices, the view provides access to.
         */
        typedef uint32 index_type;

        /**
         * The type of the values, the view provides access to.
         */
        typedef int32 value_type;

        /**
         * An iterator that provides read-only access to all nominal values.
         */
        typedef const int32* value_const_iterator;

        /**
         * An iterator that provides access to all nominal values and allows to modify them.
         */
        typedef int32* value_iterator;

        /**
         * An iterator that provides read-only access to the indices of the examples that are associated with each
         * nominal value, except for the majority value.
         */
        typedef const uint32* index_const_iterator;

        /**
         * An iterator that provides access to the indices of the examples that are associated with each nominal value,
         * except for the majority value, and allows to modify them.
         */
        typedef uint32* index_iterator;

        /**
         * Returns a `value_const_iterator` to the beginning of the nominal values.
         *
         * @return A `value_const_iterator` to the beginning
         */
        value_const_iterator values_cbegin() const;

        /**
         * Returns a `value_const_iterator` to the end of the nominal values.
         *
         * @return A `value_const_iterator` to the end
         */
        value_const_iterator values_cend() const;

        /**
         * Returns a `value_iterator` to the beginning of the nominal values.
         *
         * @return A `value_iterator` to the beginning
         */
        value_iterator values_begin();

        /**
         * Returns a `value_iterator` to the end of the nominal values.
         *
         * @return A `value_iterator` to the end
         */
        value_iterator values_end();

        /**
         * Returns an `index_const_iterator` to the beginning of the indices of the examples that are associated with a
         * specific nominal value.
         *
         * @param index The index of the nominal value
         * @return      An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin(uint32 index) const;

        /**
         * Returns an `index_const_iterator` to the end of the indices of the examples that are associated with a
         * specific nominal value.
         *
         * @param index The index of the nominal value
         * @return      An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend(uint32 index) const;

        /**
         * Returns an `index_iterator` to the beginning of the indices of the examples that are associated with a
         * specific nominal value.
         *
         * @param index The index of the nominal value
         * @return      An `index_iterator` to the beginning
         */
        index_iterator indices_begin(uint32 index);

        /**
         * Returns an `index_iterator` to the end of the indices of the examples that are associated with a specific
         * nominal value.
         *
         * @param index The index of the nominal value
         * @return      An `index_iterator` to the end
         */
        index_iterator indices_end(uint32 index);

        /**
         * Releases the ownership of the array that stores all nominal values. As a result, the behavior of this view
         * becomes undefined and it should not be used anymore. The caller is responsible for freeing the memory that is
         * occupied by the array.
         *
         * @return A pointer to an array that stores all nominal values
         */
        value_type* releaseValues();

        /**
         * Releases the ownership of the array that stores the indices of all examples not associated with the majority
         * value. As a result, the behavior of this view becomes undefined and it should not be used anymore. The caller
         * is responsible for freeing the memory that is occupied by the array.
         *
         * @return A pointer to the array that stores the indices of all examples not associated with the majority value
         */
        index_type* releaseIndices();

        /**
         * Releases the ownership of the array that stores the indices of the first element in `indices` that
         * corresponds to a certain value in `values`. As a result, the behavior of this view becomes undefined and it
         * should not be used anymore. The caller is responsible for freeing the memory that is occupied by the array.
         *
         * @return  A pointer to an array that stores the indices of the first element in `indices` that corresponds to
         *          a certain value in `values`
         */
        index_type* releaseIndptr();
};
