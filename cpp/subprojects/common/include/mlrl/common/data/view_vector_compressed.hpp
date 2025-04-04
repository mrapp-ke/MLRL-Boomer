/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_compressed.hpp"

/**
 * A view that provides access to indices that have been grouped into bins.
 */
class MLRLCOMMON_API CompressedVector : public CompressedView {
    public:

        /**
         * The number of bins.
         */
        uint32 numBins;

        /**
         * @param indices   A pointer to an array of type `uint32`, shape `(numIndices)` that stores indices
         * @param indptr    A pointer to an array that stores the indices of the first element in `indices` that
         *                  corresponds to a certain bin
         * @param numBins   The number of bins
         */
        CompressedVector(uint32* indices, uint32* indptr, uint32 numBins)
            : CompressedView(indices, indptr), numBins(numBins) {}

        /**
         * @param other A reference to an object of type `CompressedVector` that should be copied
         */
        CompressedVector(const CompressedVector& other)
            : CompressedVector(other.indices, other.indptr, other.numBins) {}

        /**
         * @param other A reference to an object of type `CompressedVector` that should be moved
         */
        CompressedVector(CompressedVector&& other) : CompressedVector(other.indices, other.indptr, other.numBins) {}

        virtual ~CompressedVector() override {}

        /**
         * An iterator that provides read-only access to the indices.
         */
        typedef const uint32* index_const_iterator;

        /**
         * An iterator that provides access to the indices and allows to modify them.
         */
        typedef uint32* index_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices that are associated with a specific bin.
         *
         * @param index The index of the nominal value
         * @return      An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin(uint32 index) const {
            return &indices[indptr[index]];
        }

        /**
         * Returns an `index_const_iterator` to the end of the indices of the examples that are associated with a
         * specific bin.
         *
         * @param index The index of the bin
         * @return      An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend(uint32 index) const {
            return &indices[indptr[index + 1]];
        }

        /**
         * Returns an `index_iterator` to the beginning of the indices of the examples that are associated with a
         * specific bin.
         *
         * @param index The index of the bin
         * @return      An `index_iterator` to the beginning
         */
        index_iterator indices_begin(uint32 index) {
            return &indices[indptr[index]];
        }

        /**
         * Returns an `index_iterator` to the end of the indices of the examples that are associated with a specific
         * bin.
         *
         * @param index The index of the bin
         * @return      An `index_iterator` to the end
         */
        index_iterator indices_end(uint32 index) {
            return &indices[indptr[index + 1]];
        }
};
