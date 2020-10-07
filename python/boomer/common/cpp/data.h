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
        virtual uint32 getNumElements() const = 0;

};

/**
 * Defines an interface for all one-dimensional vectors that provide random access to their elements.
 *
 * @tparam T The type of the data that is stored in the vector
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
        virtual T getValue(uint32 pos) const = 0;

};

/**
 * Defines an interface for all one-dimensional vectors that provide random access to indices.
 */
class IIndexVector : virtual public IVector {

    public:

        virtual ~IIndexVector() { };

        /**
         * Returns the index at a specific position.
         *
         * @param pos   The position of the index. Must be in [0, getNumElements())
         * @return      The index at the given position
         */
        virtual uint32 getIndex(uint32 pos) const = 0;

};

/**
 * An one-dimensional vector that provides random access to a fixed number of elements stored in a C-contiguous array.
 *
 * @tparam The type of the data that is stored in the vector
 */
template<class T>
class DenseVector : virtual public IRandomAccessVector<T> {

    private:

        T* array_;

        uint32 numElements_;

    public:

        /**
         * @param numElements The number of elements in the vector. Must be at least 1
         */
        DenseVector(uint32 numElements);

        /**
         * @param numElements   The number of elements in the vector. Must be at least 1
         * @param allZero       True, if all elements in the vector should be value-initialized, false otherwise
         */
        DenseVector(uint32 numElements, bool init);

        ~DenseVector();

        typedef T* iterator;

        typedef const T* const_iterator;

        /**
         * Returns an `iterator` to the beginning of the vector.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin();

        /**
         * Returns an `iterator` to the end of the vector.
         *
         * @return An `iterator` to the end
         */
        iterator end();

        /**
         * Returns a `const_iterator` to the beginning of the vector.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the vector.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const;

        uint32 getNumElements() const override;

        T getValue(uint32 pos) const override;

};

/**
 * An one-dimensional vector that provides random access to a fixed number of indices stored in a C-contiguous array.
 */
class DenseIndexVector : virtual public IIndexVector {

    private:

        const uint32* indices_;

        uint32 numElements_;

    public:

        /**
         * @param indices       A pointer to an array of type `uint32`, shape `(numElements)`, that stores the indices
         * @param numElements   The number of elements in the vector. Must be at least 1
         */
        DenseIndexVector(const uint32* indices, uint32 numElements);

        ~DenseIndexVector();

        uint32 getNumElements() const override;

        uint32 getIndex(uint32 pos) const override;

};

/**
 * An one-dimensional vector that provides random access to all indices within a continuous range [0, numIndices).
 */
class RangeIndexVector : virtual public IIndexVector {

    private:

        uint32 numIndices_;

    public:

        /**
         * @param numIndices The number of indices, the vector provides access to. Must be at least 1
         */
        RangeIndexVector(uint32 numIndices);

        uint32 getNumElements() const override;

        uint32 getIndex(uint32 pos) const override;

};

/**
 * A sparse vector that stores binary data using the dictionary of keys (DOK) format.
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

        uint32 getNumElements() const override;

        uint8 getValue(uint32 pos) const override;

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
        virtual uint32 getNumRows() const = 0;

        /**
         * Returns the number of columns in the matrix.
         *
         * @return The number of columns
         */
        virtual uint32 getNumCols() const = 0;

};

/**
 * Defines an interface for all two-dimensional matrices that provide random access to their elements.
 *
 * @tparam The type of the data that is stored in the matrix
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
        virtual T getValue(uint32 row, uint32 col) const = 0;

};

/**
 * A sparse matrix that stores binary data using the dictionary of keys (DOK) format.
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

        uint8 getValue(uint32 row, uint32 col) const override;

        uint32 getNumRows() const override;

        uint32 getNumCols() const override;

};
