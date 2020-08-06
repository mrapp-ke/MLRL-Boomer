"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides a function that allows to create a wrapper for executing different LAPACK routines.

The function pointers to the different LAPACK routines are initialized such that they refer to the functions provided by
scipy.
"""
from boomer.common._arrays cimport float64

from scipy.linalg.cython_lapack cimport dsysv


cdef extern from "cpp/lapack.h" nogil:

    ctypedef void (*dsysv_t)(char* uplo, int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, double* work, int* lwork, int* info)

    cdef cppclass Lapack:

        # Constructors:

        Lapack(dsysv_t dsysvFunction) except +

        # Functions:

        int queryDsysvLworkParameter(float64* tmpArray1, float64* output, int n) except +

        void dsysv(const float64* coefficients, const float64* invertedOrdinates, float64* tmpArray1, int* tmpArray2,
                   double* tmpArray3, float64* output, int n, int lwork, float64 l2RegularizationWeight) except +


cdef inline Lapack* init_lapack():
    """
    Creates a new wrapper for executing different LAPACK routines.

    :return: A pointer an object of type `Lapack` that allows to execute different LAPACK routines
    """
    return new Lapack(dsysv)
