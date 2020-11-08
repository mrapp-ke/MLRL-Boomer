/**
 * Provides classes that provide access to data that is stored in matrices or vectors.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include "tuples.h"
#include <unordered_set>
#include <utility>


/**
 * An one-dimensional vector that provides random access to a fixed number of elements stored in a C-contiguous array.
 *
 * @tparam T The type of the data that is stored in the vector
 */
template<class T>
class DenseVector {

    protected:

        T* array_;

        uint32 numElements_;

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        DenseVector(uint32 numElements);

        /**
         * @param numElements   The number of elements in the vector
         * @param init          True, if all elements in the vector should be value-initialized, false otherwise
         */
        DenseVector(uint32 numElements, bool init);

        /**
         * @param vector A reference to an object of type `DenseVector` to be copied
         */
        DenseVector(const DenseVector<T>& vector);

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

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements in the vector
         */
        uint32 getNumElements() const;

        /**
         * Sets the number of elements in the vector.
         *
         * @param numElements The number of elements to be set
         */
        void setNumElements(uint32 numElements);

        /**
         * Returns the value of the element at a specific position.
         *
         * @param pos   The position of the element. Must be in [0, getNumElements())
         * @return      The value of the given element
         */
        T getValue(uint32 pos) const;

};

/**
 * An one-dimensional sparse vector that stores a fixed number of elements, consisting of an index and a value, in a
 * C-contiguous array.
 *
 * @tparam T The type of the data that is stored in the vector
 */
template<class T>
class SparseArrayVector {

    public:

        typedef IndexedValue<T> Entry;


    private:

        Entry* array_;

        uint32 numElements_;

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        SparseArrayVector(uint32 numElements);

        ~SparseArrayVector();

        typedef Entry* iterator;

        typedef const Entry* const_iterator;

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

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements in the vector
         */
        uint32 getNumElements() const;

        /**
         * Sets the number of elements in the vector.
         *
         * @param numElements The number of elements to be set
         */
        void setNumElements(uint32 numElements);

        /**
         * Sorts the elements in the vector in ascending order based on their values.
         */
        void sortByValues();

};

/**
 * An one-dimensional sparse vector that stores binary data using the dictionary of keys (DOK) format.
 */
class BinaryDokVector {

    private:

        std::unordered_set<uint32> data_;

    public:

        typedef std::unordered_set<uint32>::const_iterator index_const_iterator;

        /**
         * Returns an `index_const_iterator` to the beginning of the indices.
         *
         * @return An `index_const_iterator` to the beginning
         */
        index_const_iterator indices_cbegin() const;

        /**
         * Returns an `index_const_iterator` to the end of the indices.
         *
         * @return An `index_const_iterator` to the end
         */
        index_const_iterator indices_cend() const;

        /**
         * Returns the value of the element at a specific position.
         *
         * @param pos   The position of the element. Must be in [0, getNumElements())
         * @return      The value of the given element
         */
        bool getValue(uint32 pos) const;

        /**
         * Sets a non-zero value to the element at a specific position.
         *
         * @param pos The position of the element. Must be in [0, getNumElements())
         */
        void setValue(uint32 pos);

        /**
         * Sets the values of all elements to zero.
         */
        void setAllToZero();

};

/**
 * A two-dimensional matrix that provides random access to a fixed number of elements stored in a C-contiguous array.
 *
 * @tparam T The type of the data that is stored in the matrix
 */
template<class T>
class DenseMatrix {

    protected:

        T* array_;

        uint32 numRows_;

        uint32 numCols_;

    public:

        /**
         * @param numRows   The number of rows in the matrix
         * @param numCols   The number of columns in the matrix
         */
        DenseMatrix(uint32 numRows, uint32 numCols);

        /**
         * @param numRows   The number of rows in the matrix
         * @param numCols   The number of columns in the matrix
         * @param init      True, if all elements in the matrix should be value-initialized, false otherwise
         */
        DenseMatrix(uint32 numRows, uint32 numCols, bool init);

        /**
         * @param matrix A reference to an object of type `DenseMatrix` to be copied
         */
        DenseMatrix(const DenseMatrix<T>& matrix);

        ~DenseMatrix();

        typedef typename DenseVector<T>::iterator iterator;

        typedef typename DenseVector<T>::const_iterator const_iterator;

        /**
         * Returns an `iterator` to the beginning of the matrix.
         *
         * @return An `iterator` to the beginning
         */
        iterator begin();

        /**
         * Returns an `iterator` to the end of the matrix.
         *
         * @return An `iterator` to the end
         */
        iterator end();

        /**
         * Returns a `const_iterator` to the beginning of the matrix.
         *
         * @return A `const_iterator` to the beginning
         */
        const_iterator cbegin() const;

        /**
         * Returns a `const_iterator` to the end of the matrix.
         *
         * @return A `const_iterator` to the end
         */
        const_iterator cend() const;

        /**
         * Returns an `iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      An `iterator` to the beginning of the given row
         */
        iterator row_begin(uint32 row);

        /**
         * Returns an `iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      An `iterator` to the end of the given row
         */
        iterator row_end(uint32 row);

        /**
         * Returns a `const_iterator` to the beginning of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the beginning of the given row
         */
        const_iterator row_cbegin(uint32 row) const;

        /**
         * Returns a `const_iterator` to the end of a specific row.
         *
         * @param row   The row
         * @return      A `const_iterator` to the end of the given row
         */
        const_iterator row_cend(uint32 row) const;

        /**
         * Returns the number of rows in the matrix.
         *
         * @return The number of rows in the matrix
         */
        uint32 getNumRows() const;

        /**
         * Returns the number of columns in the matrix.
         *
         * @return The number of columns in the matrix
         */
        uint32 getNumCols() const;

};

/**
 * A two-dimensional sparse matrix that stores binary data using the dictionary of keys (DOK) format.
 */
class BinaryDokMatrix {

    private:

        typedef std::pair<uint32, uint32> Entry;

        /**
         * Implements a hash function for elements of type `Entry`..
         */
        struct HashFunction {

            inline std::size_t operator()(const Entry &v) const {
                return (((uint64) v.first) << 32) | ((uint64) v.second);
            }

        };

        std::unordered_set<Entry, HashFunction> data_;

    public:

        /**
         * Returns the value of the element at a specific position.
         *
         * @param row   The row of the element. Must be in [0, getNumRows())
         * @param col   The column of the element. Must be in [0, getNumCols())
         * @return      The value of the given element
         */
        bool getValue(uint32 row, uint32 col) const;

        /**
         * Sets a non-zero value to the element at a specific position.
         *
         * @param row       The row of the element. Must be in [0, getNumRows())
         * @param column    The column of the element. Must be in [0, getNumCols())
         */
        void setValue(uint32 row, uint32 column);

        /**
         * Sets the values of all elements to zero.
         */
        void setAllToZero();

};
