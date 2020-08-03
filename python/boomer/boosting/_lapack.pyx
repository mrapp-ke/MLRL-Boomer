"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Initializes the function pointers to different LAPACK routines (defined in `cpp/lapack.h`).
"""
from scipy.linalg.cython_lapack cimport dsysv


cdef extern from "cpp/lapack.h":

    ctypedef void (*dsysv_t)(char* uplo, int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, double* work, int* lwork, int* info)

    dsysv_t dsysvFunction


# Set the function pointer to the DSYSV routine provided by scipy
dsysvFunction = dsysv
