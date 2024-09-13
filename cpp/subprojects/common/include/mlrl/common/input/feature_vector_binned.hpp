/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view.hpp"

/**
 * A feature vector that stores the indices of the examples that are associated with each bin, except for the most
 * frequent one, created by a method that assigns numerical feature values to bins.
 */
class MLRLCOMMON_API BinnedFeatureVector {
    public:

        /**
         * A pointer to an array that stores thresholds separating adjacent bins.
         */
        float32* thresholds;

        /**
         * A pointer to an array that stores the indices of all examples not associated with the most frequent bin.
         */
        uint32* indices;

        /**
         * A pointer to an array that stores the indices of the first element in `indices` that corresponds to a certain
         * bin.
         */
        uint32* indptr;

        /**
         * The number of bins, excluding the most frequent one.
         */
        uint32 numBins;

        /**
         * The index of the most frequent bin.
         */
        uint32 sparseBinIndex;

        /**
         * @param thresholds        A pointer to an array of type `float32`, shape `(numBins - 1)` that stores
         *                          thresholds separating bins
         * @param indices           A pointer to an array of type `uint32`, shape `(numIndices)` that stores the indices
         *                          of all examples not associated with the most frequent bin
         * @param indptr            A pointer to an array that stores the indices of the first element in `indices` that
         *                          corresponds to a certain bin
         * @param numBins           The number of bins, including the most frequent one
         * @param numIndices        The number of elements in the array `indices`
         * @param sparseBinIndex    The index of the most frequent bin
         */
        BinnedFeatureVector(float32* thresholds, uint32* indices, uint32* indptr, uint32 numBins, uint32 numIndices,
                            uint32 sparseBinIndex);

        /**
         * @param other A reference to an object of type `BinnedFeatureVector` that should be copied
         */
        BinnedFeatureVector(const BinnedFeatureVector& other);

        /**
         * @param other A reference to an object of type `BinnedFeatureVector` that should be moved
         */
        BinnedFeatureVector(BinnedFeatureVector&& other);

        virtual ~BinnedFeatureVector() {}

        /**
         * The type of the indices, the view provides access to.
         */
        typedef uint32 index_type;

        /**
         * The type of the thresholds, the view provides access to.
         */
        typedef float32 threshold_type;

        /**
         * An iterator that provides read-only access to all thresholds.
         */
        typedef const float32* threshold_const_iterator;

        /**
         * An iterator that provides access to all thresholds and allows to modify them.
         */
        typedef float32* threshold_iterator;

        /**
         * An iterator that provides read-only access to the indices of the examples that are associated with each bin,
         * except for the most frequent bin.
         */
        typedef const uint32* index_const_iterator;

        /**
         * An iterator that provides access to the indices of the examples that are associated with each bin, except for
         * the most frequent bin, and allows to modify them.
         */
        typedef uint32* index_iterator;

        /**
         * Returns a `threshold_const_iterator` to the beginning of the thresholds.
         *
         * @return A `threshold_const_iterator` to the beginning
         */
        threshold_const_iterator thresholds_cbegin() const;

        /**
         * Returns a `value_const_iterator` to the end of the thresholds.
         *
         * @return A `value_const_iterator` to the end
         */
        threshold_const_iterator thresholds_cend() const;

        /**
         * Returns a `value_iterator` to the beginning of the thresholds.
         *
         * @return A `value_iterator` to the beginning
         */
        threshold_iterator thresholds_begin();

        /**
         * Returns a `threshold_iterator` to the end of the thresholds.
         *
         * @return A `threshld_iterator` to the end
         */
        threshold_iterator thresholds_end();

        /**
         * Returns an `index_const_iterator` to the beginning of the indices of the examples that are associated with a
         * specific bin.
         *
         * @param index The index of the bin
         * @return      An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin(uint32 index) const;

        /**
         * Returns an `index_const_iterator` to the end of the indices of the examples that are associated with a
         * specific bin.
         *
         * @param index The index of the bin
         * @return      An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend(uint32 index) const;

        /**
         * Returns an `index_iterator` to the beginning of the indices of the examples that are associated with a
         * specific bin.
         *
         * @param index The index of the bin
         * @return      An `index_iterator` to the beginning
         */
        index_iterator indices_begin(uint32 index);

        /**
         * Returns an `index_iterator` to the end of the indices of the examples that are associated with a specific
         * bin.
         *
         * @param index The index of the bin
         * @return      An `index_iterator` to the end
         */
        index_iterator indices_end(uint32 index);

        /**
         * Releases the ownership of the array that stores the thresholds. As a result, the behavior of this view
         * becomes undefined and it should not be used anymore. The caller is responsible for freeing the memory that is
         * occupied by the array.
         *
         * @return A pointer to an array that stores all thresholds
         */
        threshold_type* releaseThresholds();

        /**
         * Releases the ownership of the array that stores the indices of all examples not associated with the most
         * frequent bin. As a result, the behavior of this view becomes undefined and it should not be used anymore. The
         * caller is responsible for freeing the memory that is occupied by the array.
         *
         * @return A pointer to the array that stores the indices of all examples not associated with the most frequent
         *         bin
         */
        index_type* releaseIndices();

        /**
         * Releases the ownership of the array that stores the indices of the first element in `indices` that
         * corresponds to a certain bin. As a result, the behavior of this view becomes undefined and it should not be
         * used anymore. The caller is responsible for freeing the memory that is occupied by the array.
         *
         * @return  A pointer to an array that stores the indices of the first element in `indices` that corresponds to
         *          a certain bin
         */
        index_type* releaseIndptr();
};
