/**
 * Implements classes that provide access to data that is stored in matrices or vectors.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"


/**
 * An abstract base class for all two-dimensional matrices.
 */
class AbstractMatrix {

    public:

        virtual ~AbstractMatrix();

        /**
         * Returns the number of rows in the matrix.
         *
         * @return The number of rows
         */
        virtual uint32 getNumRows();

        /**
         * Returns the number of columns in the matrix.
         *
         * @return The number of columns
         */
        virtual uint32 getNumCols();

};

/**
 * An abstract base class for all two-dimensional matrices that provide random access to their elements.
 */
template<class T>
class AbstractRandomAccessMatrix : public AbstractMatrix {

    public:

        /**
         * Returns the element at a specific position.
         *
         * @param row   The row of the element to be returned
         * @param col   The column of the element to be returned
         * @return      The element at the given position
         */
        virtual T get(uint32 row, uint32 col);

};

/**
 * An abstract base class for all one-dimensional vectors.
 */
class AbstractVector {

    public:

        virtual ~AbstractVector();

        /**
         * Returns the number of elements in the vector.
         *
         * @return The number of elements
         */
        virtual uint32 getNumElements();

};
