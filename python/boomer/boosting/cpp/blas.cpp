#include "blas.h"

ddot_t ddotFunction;


float64 ddot(float64* x, float64* y, int n) {
    // Storage spacing between the elements of the arrays x and y
    int inc = 1;
    // Invoke the DDOT routine...
    return ddotFunction(&n, x, &inc, y, &inc);
}

cdef inline float64* __dspmv_float64(float64* a, float64* x, int n):
    # 'U' if the upper-right triangle of A should be used, 'L' if the lower-left triangle should be used
    cdef char* uplo = 'U'
    # A scalar to be multiplied with the matrix A
    cdef double alpha = 1
    # The increment for the elements of x
    cdef int incx = 1
    # A scalar to be multiplied with vector y
    cdef double beta = 0
    # An array of type `float64`, shape `(n)`. Will contain the result of A * x
    cdef float64* y = <float64*>malloc(n * sizeof(float64))
    # The increment for the elements of y
    cdef int incy = 1
    # Invoke the DSPMV routine...
    dspmv(uplo, &n, &alpha, a, x, &incx, &beta, y, &incy)
    return y
