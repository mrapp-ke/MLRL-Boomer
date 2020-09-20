/**
 * Provides implementations of sparse matrices.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include "data.h"
#include <unordered_set>
#include <utility>


/**
 * Implements a hash function for pairs that store two integers of type `uint32`.
 */
struct PairHash {

    inline std::size_t operator()(const std::pair<uint32, uint32> &v) const {
        return (((uint64) v.first) << 32) | ((uint64) v.second);
    }

};

/**
 * A sparse matrix that stores binary values using the dictionary of keys (DOK) format.
 */
class BinaryDokMatrix : virtual public IRandomAccessMatrix<uint8> {

    private:

        std::unordered_set<std::pair<uint32, uint32>, PairHash> data_;

    public:

        /**
         * Sets the element at a specific position to a non-zero value.
         *
         * @param row       The row of the element to be set
         * @param column    The column of the element to be set
         */
        void addValue(uint32 row, uint32 column);

        uint8 get(uint32 row, uint32 col) override;

};

/**
 * A sparse vector that stores binary values using the dictionary of keys (DOK) format.
 */
class BinaryDokVector : virtual public IRandomAccessVector<uint8> {

    private:

        std::unordered_set<uint32> data_;

    public:

        /**
         * Sets the element at a specific position to non-zero value.
         *
         * @param pos The position of the element to be set
         */
        void addValue(uint32 pos);

        /**
         * Returns the element at a specific position.
         *
         * @param pos   The position of the element to be returned
         * @return      The element at the given position
         */
        uint8 get(uint32 pos) override;

};
