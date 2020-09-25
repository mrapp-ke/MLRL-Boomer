/**
 * Provides interfaces and classes that provide access to data that is stored in matrices or vectors.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
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
 * Defines an interface for all one-dimensional vectors.
 */
class IVector {

    public:

        virtual ~IVector() { };

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements
         */
        virtual uint32 getNumElements() = 0;

};

/**
 * Defines an interface for all one-dimensional vectors that provide random access to their elements.
 */
template<class T>
class IRandomAccessVector : virtual public IVector {

    public:

        virtual ~IRandomAccessVector() { };

        /**
         * Returns the value of the element at a specific position.
         *
         * @param pos   The position of the element. Must be in [0, getNumElements())
         * @return      The value of the given element
         */
        virtual T getValue(uint32 pos) = 0;

};

/**
 * A sparse vector that stores binary values using the dictionary of keys (DOK) format.
 */
class BinaryDokVector : virtual public IRandomAccessVector<uint8> {

    private:

        uint32 numElements_;

        std::unordered_set<uint32> data_;

    public:

        /**
         * @param numElements The number of elements in the vector. Must be at least 1
         */
        BinaryDokVector(uint32 numElements);

        /**
         * Sets a non-zero value to the element at a specific position.
         *
         * @param pos The position of the element. Must be in [0, getNumElements())
         */
        void setValue(uint32 pos);

        uint8 getValue(uint32 pos) override;

        uint32 getNumElements() override;

};

/**
 * Defines an interface for all two-dimensional matrices.
 */
class IMatrix {

    public:

        virtual ~IMatrix() { };

        /**
         * Returns the number of rows in the matrix.
         *
         * @return The number of rows
         */
        virtual uint32 getNumRows() = 0;

        /**
         * Returns the number of columns in the matrix.
         *
         * @return The number of columns
         */
        virtual uint32 getNumCols() = 0;

};

/**
 * Defines an interface for all two-dimensional matrices that provide random access to their elements.
 */
template<class T>
class IRandomAccessMatrix : virtual public IMatrix {

    public:

        virtual ~IRandomAccessMatrix() { };

        /**
         * Returns the value of the element at a specific position.
         *
         * @param row   The row of the element. Must be in [0, getNumRows())
         * @param col   The column of the element. Must be in [0, getNumCols())
         * @return      The value of the given element
         */
        virtual T getValue(uint32 row, uint32 col) = 0;

};

/**
 * A sparse matrix that stores binary values using the dictionary of keys (DOK) format.
 */
class BinaryDokMatrix : virtual public IRandomAccessMatrix<uint8> {

    private:

        uint32 numRows_;

        uint32 numCols_;

        std::unordered_set<std::pair<uint32, uint32>, PairHash> data_;

    public:

        /**
         * @param numRows   The number of rows in the matrix. Must be at least 1
         * @param numCols   The number of columns in the matrix. Must be at least 1
         */
        BinaryDokMatrix(uint32 numRows, uint32 numCols);

        /**
         * Sets a non-zero value to the element at a specific position.
         *
         * @param row       The row of the element. Must be in [0, getNumRows())
         * @param column    The column of the element. Must be in [0, getNumCols())
         */
        void setValue(uint32 row, uint32 column);

        uint8 getValue(uint32 row, uint32 col) override;

        uint32 getNumRows() override;

        uint32 getNumCols() override;

};
