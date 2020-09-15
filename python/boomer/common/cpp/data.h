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
