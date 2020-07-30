/**
 * Provides type definitions using names that are consistent to those used in `arrays.pxd`.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include <cstdint>
#include <stdlib.h>

typedef uint8_t uint8;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef intptr_t intp;
typedef double float64;


namespace arrays {

    /**
     * Allocates and returns a new C-contiguous array of type `float64`, shape `(numElements)`.
     *
     * @param numElements   The number of elements in the array
     * @return              A pointer to the array that has been allocated
     */
    static inline float64* mallocFloat64(intp numElements) {
        return (float64*) malloc(numElements * sizeof(float64));
    }

    /**
     * Allocates and returns a new C-contiguous array of type `float64`, shape `(numRows, numCols)`.
     *
     * @param numRows   The number of rows in the array
     * @param numCols   The number of columns in the array
     * @return          A pointer to the array that has been allocated
     */
    static inline float64* mallocFloat64(intp numRows, intp numCols) {
        return (float64*) malloc(numRows * numCols * sizeof(float64));
    }

    /**
     * Sets all elements in an array to zero.
     *
     * @param a             A pointer to an array of template type `T`, shape `(numRows, numCols)`
     * @param numElements   The number of elements in the array
     */
    template<typename T>
    static inline void setToZeros(T* a, intp numRows, intp numCols) {
        for (intp i = 0; i < numRows * numCols; i++) {
            a[i] = 0;
        }
    }

}
