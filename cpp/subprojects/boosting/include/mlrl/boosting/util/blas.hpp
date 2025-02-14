/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to execute BLAS routines.
     *
     * @tparam T The type of the arrays on which the BLAS routines should operate
     */
    template<typename T>
    class Blas final {
        public:

            /**
             * A function pointer to BLAS' DOT routine.
             */
            typedef T (*DotFunction)(int* n, T* x, int* incx, T* y, int* incy);

            /**
             * A function pointer to BLAS' SPMV routine.
             */
            typedef void (*SpmvFunction)(char* uplo, int* n, T* alpha, T* ap, T* x, int* incx, T* beta, T* y,
                                         int* incy);

            /**
             * A struct that stores function pointers to all supported BLAS routines.
             */
            struct Routines {
                public:

                    /**
                     * A function pointer to BLAS' DOT routine.
                     */
                    const DotFunction dot;

                    /**
                     *  A function pointer to BLAS' SPMV routine.
                     */
                    const SpmvFunction spmv;

                    /**
                     * @param dot   A function pointer to BLAS' DOT routine
                     * @param spmv  A function pointer to BLAS' SPMV routine
                     */
                    Routines(DotFunction dot, SpmvFunction spmv) : dot(dot), spmv(spmv) {}
            };

        private:

            const DotFunction dot_;

            const SpmvFunction spmv_;

        public:

            /**
             * @param routines A reference to an object of type `Blas::Routines` that stores function pointers to all
             *                 supported BLAS routines
             */
            Blas(const Routines& routines);

            /**
             * Computes and returns the dot product x * y of two vectors x and y using BLAS' DOT routine (see
             * https://www.netlib.org/lapack/explore-html/d1/dcc/group__dot.html).
             *
             * @param x A pointer to an array of template type `T`, shape `(n)`, representing the first vector x
             * @param y A pointer to an array of template type `T`, shape `(n)`, representing the second vector y
             * @param n The number of elements in the arrays `x` and `y`
             * @return  A scalar of template type `T`, representing the result of the dot product x * y
             */
            T dot(T* x, T* y, int n) const;

            /**
             * Computes and returns the solution to the matrix-vector operation A * x using BLAS' SPMV routine (see
             * https://www.netlib.org/lapack/explore-html/d0/d4b/group__hpmv.html).
             *
             * SPMV expects the matrix A to be a symmetric matrix with shape `(n, n)` and x to be an array with shape
             * `(n)`. The matrix A must be supplied in packed form, i.e., as an array with shape `(n * (n + 1) / 2 )`
             * that consists of the columns of A appended to each other and omitting all unspecified elements.
             *
             * @param a         A pointer to an array of template type `T`, shape `(n * (n + 1) / 2)`, representing the
             *                  elements in the upper-right triangle of the matrix A in a packed form
             * @param x         A pointer to an array of template type `T`, shape `(n)`, representing the elements in
             *                  the array x
             * @param output    A pointer to an array of template type `T`, shape `(n)`, the result of the matrix-vector
             *                  operation A * x should be written to. May contain arbitrary values
             * @param n         The number of elements in the arrays `a` and `x`
             */
            void spmv(T* a, T* x, T* output, int n) const;
    };

    /**
     * A factory that allows to create instances of type `Blas`.
     */
    class BlasFactory final {
        private:

            const Blas<float32>::Routines float32Routines_;

            const Blas<float64>::Routines float64Routines_;

        public:

            /**
             * @param float32Routines   A reference to an object of type `Blas::Routines` that stores function pointers
             *                          to all supported BLAS routines operating of 32-bit floating point values
             * @param float64Routines   A reference to an object of type `Blas::Routines` that stores function pointers
             *                          to all supported BLAS routines operating of 64-bit floating point values
             */
            BlasFactory(const Blas<float32>::Routines& float32Routines, const Blas<float64>::Routines& float64Routines);

            /**
             * Create and return an object of type `Blas` that allows to execute BLAS routines operating on 32-bit
             * floating point arrays.
             *
             * @return An unique pointer to an object of type `Blas` that has been created
             */
            std::unique_ptr<Blas<float32>> create32Bit() const;

            /**
             * Create and return an object of type `Blas` that allows to execute BLAS routines operating on 64-bit
             * floating point arrays.
             *
             * @return An unique pointer to an object of type `Blas` that has been created
             */
            std::unique_ptr<Blas<float64>> create64Bit() const;
    };

}
