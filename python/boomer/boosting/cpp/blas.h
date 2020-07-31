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
