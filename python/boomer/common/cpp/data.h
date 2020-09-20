/**
 * Provides interfaces and classes that provide access to data that is stored in matrices or vectors.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"


/**
 * Defines an interface for all two-dimensional matrices.
 */
class IMatrix {

    public:

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

        /**
         * Returns the element at a specific position.
         *
         * @param row   The row of the element to be returned
         * @param col   The column of the element to be returned
         * @return      The element at the given position
         */
        virtual T get(uint32 row, uint32 col) = 0;

};

/**
 * Defines an interface for all one-dimensional vectors.
 */
class IVector {

    public:

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

        /**
         * Returns the element at a specific position.
         *
         * @param pos   The position of the element to be returned
         * @return      The element at the given position
         */
        virtual T get(uint32 pos) = 0;

};
