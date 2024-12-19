/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

namespace boosting {

    /**
     * Allows to execute BLAS routines.
     */
    class Blas final {
        public:

            /**
             * A function pointer to BLAS' DDOT routine.
             */
            typedef double (*DdotFunction)(int* n, double* dx, int* incx, double* dy, int* incy);

            /**
             * A function pointer to BLAS' DSPMV routine.
             */
            typedef void (*DspmvFunction)(char* uplo, int* n, double* alpha, double* ap, double* x, int* incx,
                                          double* beta, double* y, int* incy);

        private:

            const DdotFunction ddotFunction_;

            const DspmvFunction dspmvFunction_;

        public:

            /**
             * @param ddotFunction  A function pointer to BLAS' DDOT routine
             * @param dspmvFunction A function pointer to BLAS' DSPMV routine
             */
            Blas(DdotFunction ddotFunction, DspmvFunction dspmvFunction);

            /**
             * Computes and returns the dot product x * y of two vectors x and y using BLAS' DDOT routine (see
             * https://www.netlib.org/lapack/explore-html/d1/dcc/group__dot_ga2a42ecc597403b22ad786715c739196b.html#ga2a42ecc597403b22ad786715c739196b).
             *
             * @param x A pointer to an array of type `float64`, shape `(n)`, representing the first vector x
             * @param y A pointer to an array of type `float64`, shape `(n)`, representing the second vector y
             * @param n The number of elements in the arrays `x` and `y`
             * @return  A scalar of type `float64`, representing the result of the dot product x * y
             */
            float64 ddot(float64* x, float64* y, int n) const;

            /**
             * Computes and returns the solution to the matrix-vector operation A * x using BLAS' DSPMV routine (see
             * https://www.netlib.org/lapack/explore-html/d0/d4b/group__hpmv_ga739f8dc2316523832bde2b237fcad8a6.html#ga739f8dc2316523832bde2b237fcad8a6).
             *
             * DSPMV expects the matrix A to be a symmetric matrix with shape `(n, n)` and x to be an array with shape
             * `(n)`. The matrix A must be supplied in packed form, i.e., as an array with shape `(n * (n + 1) / 2 )`
             * that consists of the columns of A appended to each other and omitting all unspecified elements.
             *
             * @param a         A pointer to an array of type `float64`, shape `(n * (n + 1) / 2)`, representing the
             *                  elements in the upper-right triangle of the matrix A in a packed form
             * @param x         A pointer to an array of type `float64`, shape `(n)`, representing the elements in the
             *                  array x
             * @param output    A pointer to an array of type `float64`, shape `(n)`, the result of the matrix-vector
             *                  operation A * x should be written to. May contain arbitrary values
             * @param n         The number of elements in the arrays `a` and `x`
             */
            void dspmv(float64* a, float64* x, float64* output, int n) const;
    };

}
