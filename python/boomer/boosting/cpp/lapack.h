/**
 * Provides wrapper functions for executing different LAPACK routines.
 *
 * The function pointers to the different LAPACK routines are supposed to be initialized externally.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/arrays.h"

// Defines a function pointer to the DSYSV routine
typedef double (*dsysv_t)(char* uplo, int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, double* work, int* lwork, int* info);
extern dsysv_t dsysvFunction;


/**
 * Computes and returns the solution to a system of linear equations A * X = B using LAPACK's DSYSV solver (see
 * http://www.netlib.org/lapack/explore-html/d6/d0e/group__double_s_ysolve_ga9995c47692c9885ed5d6a6b431686f41.html).
 *
 * DSYSV requires A to be a matrix with shape `(n, n)`, representing the coefficients, and B to be a matrix with shape
 * `(n, nrhs)`, representing the ordinates. X is a matrix of unknowns with shape `(n, nrhs)`.
 *
 * DSYSV will overwrite the matrices A and B. When terminated successfully, B will contain the solution to the system
 * of linear equations. To retain their state, this function will copy the given arrays before invoking DSYSV.
 *
 * Furthermore, DSYSV assumes the matrix of coefficients A to be symmetrical, i.e., it will only use the upper-right
 * triangle of A, whereas the remaining elements are ignored. For reasons of space efficiency, this function expects
 * the coefficients to be given as an array with shape `n * (n + 1) / 2`, representing the elements of the upper-right
 * triangle of A, where the columns are appended to each other and unspecified elements are omitted. This function will
 * implicitly convert the given array into a matrix that is suited for DSYSV.
 *
 * Optionally, this function allows to specify a weight to be used for L2 regularization. The given weight is added to
 * each element on the diagonal of the matrix of coefficients A.
 *
 * @param coefficients              An array of dtype `float64`, shape `n * (n + 1) / 2)`, representing the coefficients
 * @param invertedOrdinates         An array of dtype `float64`, shape `(n)`, representing the inverted ordinates, i.e.,
 *                                  the ordinates multiplied by -1. The sign of the elements in this array will be
 *                                  inverted to when creating the matrix B
 * @param n                         The number of equations
 * @param l2RegularizationWeight    A scalar of dtype `float64`, representing the weight of the L2 regularization
 * @return                          A pointer to an array of type `float64`, shape `(n)`, representing the solution to
 *                                  the system of linear equations
 */
float64* dsysv(float64* coefficients, float64* invertedOrdinates, int n, float64 l2RegularizationWeight);
