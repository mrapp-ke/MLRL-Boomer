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
 * Defines an interface for all one-dimensional, potentially sparse, vectors.
 */
class ISparseVector : virtual public IVector {

    public:

        virtual ~ISparseVector() { };

        /**
         * Returns whether the vector contains any zero elements or not.
         *
         * @return True, if the vector contains any zero elements, false otherwise
         */
        virtual bool hasZeroElements() = 0;

};

/**
 * Defines an interface for all one-dimensional, potentially sparse, vectors that provide random access to indices.
 */
class IIndexVector : virtual public ISparseVector {

    public:

        virtual ~IIndexVector() { };

        /**
         * Returns the index at a specific position.
         *
         * @param pos   The position of the index. Must be in [0, getNumElements())
         * @return      The index at the given position
         */
        virtual uint32 getIndex(uint32 pos) = 0;

};

/**
 * Defines an interface for all one-dimensional, potentially sparse, vectors that provide random access to all of their
 * elements, including zero elements that are not explicitly stored in the vector.
 */
template<class T>
class ISparseRandomAccessVector : virtual public ISparseVector, virtual public IRandomAccessVector<T> {

    public:

        virtual ~ISparseRandomAccessVector() { };

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

        uint32 getNumElements() override;

        bool hasZeroElements() override;

        uint32 getIndex(uint32 pos) override;

};

/**
 * An one-dimensional vector that provides random access to a fixed number of indices stored in a C-contiguous array.
 */
class DenseIndexVector : virtual public IIndexVector {

    private:

        uint32 numElements_;

        uint32* indices_;

    public:

        /**
         * @param numElements The number of elements in the vector. Must be at least 1
         */
        DenseIndexVector(uint32 numElements);

        ~DenseIndexVector();

        /**
         * Sets the index at a specific position.
         *
         * @param pos   The position of the index. Must be in [0, getNumElements())
         * @param index The index to be set. Must be at least 0
         */
        void setIndex(uint32 pos, uint32 index);

        uint32 getNumElements() override;

        bool hasZeroElements() override;

        uint32 getIndex(uint32 pos) override;

};

/**
 * An one-dimensional vector that stores numerical data in a C-contiguous array.
 */
template<class T>
class DenseVector : virtual public IRandomAccessVector<T> {

    private:

        uint32 numElements_;

        T* data_;

    public:

        /**
         * @param numElements The number of elements in the vector. Must be at least 1
         */
        DenseVector(uint32 numElements);

        /**
         * @param numElements   The number of elements in the vector. Must be at least 1
         * @param allZero       True, if all elements should be set to zero, false otherwise
         */
        DenseVector(uint32 numElements, bool allZero);

        ~DenseVector();

        /**
         * Sets the value of the element at a specific position.
         *
         * @param pos   The position of the element. Must be in [0, getNumElements())
         * @param value The value to be set
         */
        void set(uint32 pos, T value);

        uint32 getNumElements() override;

        T get(uint32 pos) override;

};

/**
 * A sparse vector that stores binary data using the dictionary of keys (DOK) format.
 */
class BinaryDokVector : virtual public ISparseRandomAccessVector<uint8> {

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

        uint32 getNumElements() override;

        bool hasZeroElements() override;

        uint8 getValue(uint32 pos) override;

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

        uint8 getValue(uint32 row, uint32 col) override;

        uint32 getNumRows() override;

        uint32 getNumCols() override;

};
