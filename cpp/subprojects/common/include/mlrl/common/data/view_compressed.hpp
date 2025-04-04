/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view.hpp"

/**
 * A view that provides access to indices that have been grouped into bins.
 */
class MLRLCOMMON_API CompressedView {
    public:

        /**
         * A pointer to an array that stores indices.
         */
        uint32* indices;

        /**
         * A pointer to an array that stores the indices of the first element in `indices` that corresponds to a certain
         * bin.
         */
        uint32* indptr;

        /**
         * @param indices   A pointer to an array of type `uint32`, shape `(numIndices)` that stores indices
         * @param indptr    A pointer to an array that stores the indices of the first element in `indices` that
         *                  corresponds to a certain bin
         */
        CompressedView(uint32* indices, uint32* indptr) : indices(indices), indptr(indptr) {}

        /**
         * @param other A reference to an object of type `CompressedView` that should be copied
         */
        CompressedView(const CompressedView& other) : CompressedView(other.indices, other.indptr) {}

        /**
         * @param other A reference to an object of type `CompressedView` that should be moved
         */
        CompressedView(CompressedView&& other) : CompressedView(other.indices, other.indptr) {}

        virtual ~CompressedView() {}

        /**
         * The type of the indices, the view provides access to.
         */
        typedef uint32 index_type;

        /**
         * Releases the ownership of the array that stores the indices. As a result, the behavior of this view becomes
         * undefined and it should not be used anymore. The caller is responsible for freeing the memory that is
         * occupied by the array.
         *
         * @return A pointer to the array that stores indices
         */
        index_type* releaseIndices() {
            index_type* ptr = indices;
            indices = nullptr;
            return ptr;
        }

        /**
         * Releases the ownership of the array that stores the indices of the first element in `indices` that
         * corresponds to a certain bin. As a result, the behavior of this view becomes undefined and it should not be
         * used anymore. The caller is responsible for freeing the memory that is occupied by the array.
         *
         * @return  A pointer to an array that stores the indices of the first element in `indices` that corresponds to
         *          a certain bin
         */
        index_type* releaseIndptr() {
            index_type* ptr = indptr;
            indptr = nullptr;
            return ptr;
        }
};
