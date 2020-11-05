/**
 * Provides classes and functions that implement operations on numeric data that is stored in matrices or vectors.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "data.h"
#include "indices.h"


/**
 * An one-dimensional vector that provides random access to a fixed number of numbers stored in a C-contiguous array.
 *
 * @tparam T The type of the numbers that are stored in the vector
 */
template<class T>
class DenseNumericVector : public DenseVector<T> {

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        DenseNumericVector(uint32 numElements);

        /**
         * @param numElements   The number of elements in the vector
         * @param init          True, if all elements in the vector should be value-initialized, false otherwise
         */
        DenseNumericVector(uint32 numElements, bool init);

        /**
         * Sets the values of all elements in the vector to zero.
         */
        void setAllToZero();

};

typedef DenseNumericVector<float64> DenseFloat64Vector;
