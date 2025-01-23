/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to execute LAPACK routines.
     *
     * @tparam T The type of the arrays on which the LAPACK routines should operate
     */
    template<typename T>
    class Lapack final {
        public:

            /**
             * A function pointer to LAPACK'S SYSV routine.
             */
            typedef void (*SysvFunction)(char* uplo, int* n, int* nrhs, T* a, int* lda, int* ipiv, T* b, int* ldb,
                                         T* work, int* lwork, int* info);

            /**
             * A struct that stores function pointers to all supported LAPACK routines.
             */
            struct Routines {
                public:

                    /**
                     * A function pointer to LAPACK' SYSV routine.
                     */
                    const SysvFunction sysv;

                    /**
                     * @param sysv A function pointer to LAPACK's SYSV routine
                     */
                    explicit Routines(SysvFunction sysv) : sysv(sysv) {}
            };

        private:

            const SysvFunction sysv_;

        public:

            /**
             * @param routines A reference to an object of type `Lapack::Routines` that stores function pointers to all
             *                 supported LAPACK routines
             */
            Lapack(const Routines& routines);

            /**
             * Determines and returns the optimal value for the parameter "lwork" as used by LAPACK'S SYSV routine.
             *
             * This function must be run before attempting to solve a linear system using the function `sysv` to
             * determine the optimal value for the parameter "lwork".
             *
             * @param tmpArray1 A pointer to an array of template type `T`, shape `(n, n)` that will be used by the
             *                  function `sysv` to temporarily store values computed by the SYSV routine. May contain
             *                  arbitrary values
             * @param output    A pointer to an array of template type `T`, shape `(n)`, the solution of the system of
             *                  linear equations should be written to by the function `sysv`. May contain arbitrary
             *                  values
             * @param n         The number of equations in the linear system to be solved by the function `sysv`
             * @return          The optimal value for the parameter "lwork"
             */
            int querySysvLworkParameter(T* tmpArray1, T* output, int n) const;

            /**
             * Computes and returns the solution to a linear system A * X = B using LAPACK's SYSV solver (see
             * https://www.netlib.org/lapack/explore-html/d8/ddb/group__hesv.html).
             *
             * The function `querySysvLworkParameter` must be run beforehand to determine the optimal value for the
             * parameter "lwork" and to allocate a temporary array depending on this value.
             *
             * SYSV requires A to be a matrix with shape `(n, n)`, representing the coefficients, and B to be a matrix
             * with shape `(n, nrhs)`, representing the ordinates. X is a matrix of unknowns with shape `(n, nrhs)`.
             *
             * SYSV will overwrite the matrices A and B. When terminated successfully, B will contain the solution to
             * the system of linear equations. To retain their state, this function will copy the given arrays before
             * invoking SYSV.
             *
             * Furthermore, SYSV assumes the matrix of coefficients A to be symmetrical, i.e., it will only use the
             * upper-right triangle of A, whereas the remaining elements are ignored. For reasons of space efficiency,
             * this function expects the coefficients to be given as an array with shape `n * (n + 1) / 2`, representing
             * the elements of the upper-right triangle of A, where the columns are appended to each other and
             * unspecified elements are omitted. This function will implicitly convert the given array into a matrix
             * that is suited for SYSV.
             *
             * @param tmpArray1                 A pointer to an array of template type `T`, shape `(n, n)` that stores
             *                                  the coefficients in the matrix A. It will be used to temporarily store
             *                                  values computed by the SYSV routine
             * @param tmpArray2                 A pointer to an array of type `int`, shape `(n)` that will be used to
             *                                  temporarily store values computed by the SYSV routine. May contain
             *                                  arbitrary values
             * @param tmpArray3                 A pointer to an array of template type `T`, shape `(lwork)` that will be
             *                                  used to temporarily store values computed by the SYSV routine. May
             *                                  contain arbitrary values
             * @param output                    A pointer to an array of template type `T`, shape `(n)` that stores the
             *                                  ordinates in the matrix A. The solution of the system of linear
             *                                  equations will be written to this array
             * @param n                         The number of equations
             * @param lwork                     The value for the parameter "lwork" to be used by the SYSV routine.
             *                                  Must have been determined using the function `querySysvLworkParameter`
             */
            void sysv(T* tmpArray1, int* tmpArray2, T* tmpArray3, T* output, int n, int lwork) const;
    };

    /**
     * A factory that allows to create instances of type `Lapack`.
     */
    class LapackFactory final {
        private:

            const Lapack<float32>::Routines float32Routines_;

            const Lapack<float64>::Routines float64Routines_;

        public:

            /**
             * @param float32Routines   A reference to an object of type `Lapack::Routines` that stores function
             *                          pointers to all supported LAPACK routines operating of 32-bit floating point
             *                          values
             * @param float64Routines   A reference to an object of type `Lapack::Routines` that stores function
             *                          pointers to all supported LAPACK routines operating of 64-bit floating point
             *                          values
             */
            LapackFactory(const Lapack<float32>::Routines& float32Routines,
                          const Lapack<float64>::Routines& float64Routines);

            /**
             * Creates and returns an object of type `Lapack` that allows to execute LAPACK routines operating on 32-bit
             * floating point arrays.
             *
             * @return An unique pointer to an object of type `Lapack` that has been created
             */
            std::unique_ptr<Lapack<float32>> create32Bit() const;

            /**
             * Creates and returns an object of type `Lapack` that allows to execute LAPACK routines operating on 64-bit
             * floating point arrays.
             *
             * @return An unique pointer to an object of type `Lapack` that has been created
             */
            std::unique_ptr<Lapack<float64>> create64Bit() const;
    };

}
