/**
 * Provides wrapper functions for executing different BLAS routines.
 *
 * The function pointers to the different BLAS routines are supposed to be initialized externally.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/arrays.h"

// Defines a function pointer to the DDOT routine
typedef double (*ddot_t)(int* n, double* dx, int* incx, double* dy, int* incy);
extern ddot_t ddotFunction;

// Defines a function pointer to the DSPMV routine
typedef void (*dspmv_t)(char* uplo, int* n, double* alpha, double* ap, double* x, int* incx, double* beta, double* y, int* incy);
extern dspmv_t dspmvFunction;


/**
 * Computes and returns the dot product x * y of two vectors x and y using BLAS' DDOT routine (see
 * http://www.netlib.org/lapack/explore-html/de/da4/group__double__blas__level1_ga75066c4825cb6ff1c8ec4403ef8c843a.html).
 *
 * @param x A pointer to an array of type `float64`, shape `(n)`, representing the first vector x
 * @param y A pointer to an array of type `float64`, shape `(n)`, representing the second vector y
 * @param n The number of elements in the arrays `x` and `y`
 * @return  A scalar of type `float64`, representing the result of the dot product x * y
 */
float64 ddot(float64* x, float64* y, int n);

/**
 * Computes and returns the solution to the matrix-vector operation A * x using BLAS' DSPMV routine (see
 * http://www.netlib.org/lapack/explore-html/d7/d15/group__double__blas__level2_gab746575c4f7dd4eec72e8110d42cefe9.html).
 *
 * DSPMV expects the matrix A to be a symmetric matrix with shape `(n, n)` and x to be an array with shape `(n)`. The
 * matrix A must be supplied in packed form, i.e., as an array with shape `(n * (n + 1) / 2 )` that consists of the
 * columns of A appended to each other and omitting all unspecified elements.
 *
 * @param a A pointer to an array of type `float64`, shape `(n * (n + 1) / 2)`, representing the elements in the
 *          upper-right triangle of the matrix A in a packed form
 * @param x A pointer to an array of type `float64`, shape `(n)`, representing the elements in the array x
 * @param n The number of elements in the arrays `a` and `x`
 * @return  A pointer to an array of type `float64`, shape `(n)`, representing the result of the matrix-vector operation
 *          A * x
 */
float64* dspmv(float64* a, float64* x, int n);
