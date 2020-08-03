"""
@author Michael Rapp (mrapp@ke.tu-darmstadt.de)

Initializes the function pointers to different BLAS routines (defined in `cpp/blas.h`).
"""
from scipy.linalg.cython_blas cimport ddot, dspmv


cdef extern from "cpp/blas.h":

    ctypedef double (*ddot_t)(int* n, double* dx, int* incx, double* dy, int* incy)

    ddot_t ddotFunction

    ctypedef void (*dspmv_t)(char* uplo, int* n, double* alpha, double* ap, double* x, int* incx, double* beta, double* y, int* incy)

    dspmv_t dspmvFunction


# Set the function pointer to the DDOT routine provided by scipy
ddotFunction = ddot

# Set the function pointer to the DSPMV routine provided by scipy
dspmvFunction = dspmv
