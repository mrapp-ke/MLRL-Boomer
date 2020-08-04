/**
 * Provides type definitions consistent to those used in `arrays.pxd`, as well as utility functions for allocating
 * arrays.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include <cstdint>

typedef uint8_t uint8;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef intptr_t intp;
typedef double float64;


namespace arrays {

    /**
     * Sets all elements in an one-dimensional array to zero.
     *
     * @param a             A pointer to an array of template type `T`, shape `(numElements)`
     * @param numElements   The number of elements in the array
     */
    template<typename T>
    static inline void setToZeros(T* a, intp numElements) {
        for (intp i = 0; i < numElements; i++) {
            a[i] = 0;
        }
    }

    /**
     * Sets all elements in a two-dimensional array to zero.
     *
     * @param a         A pointer to an array of template type `T`, shape `(numRows, numCols)`
     * @param numRows   The number of rows in the array
     * @param numCols   The number of columns in the array
     */
    template<typename T>
    static inline void setToZeros(T* a, intp numRows, intp numCols) {
        for (intp i = 0; i < numRows * numCols; i++) {
            a[i] = 0;
        }
    }

}
