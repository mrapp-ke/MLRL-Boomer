# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides utility functions for common mathematical operations.
"""
from boomer.algorithm._arrays cimport intp, float32, float64, matrix_float64

from scipy.linalg.cython_lapack cimport dsysv
from libc.stdlib cimport malloc, free


cdef inline divide_or_zero_float64(float64 a, float64 b):
    """
    Divides a scalar of dtype `float64` by another one. The division by zero evaluates to 0 by definition.

    :param a:   The scalar to be divided
    :param b:   The divisor
    :return:    A scalar of dtype `float64`, representing the result of a / b or 0, if b = 0
    """
    if b == 0:
        return 0
    else:
        return a / b


cdef inline float64[::1] dsysv_float64(float64[::1] coefficients, float64[::1] ordinates):
    """
    Computes and returns the solution to a system of linear equations A * X = B using LAPACK's DSYSV solver (see
    http://www.netlib.org/lapack/explore-html/d6/d0e/group__double_s_ysolve_ga9995c47692c9885ed5d6a6b431686f41.html).
    DSYSV requires A to be a double-precision matrix with shape `(num_equations, num_equations)`, representing the
    coefficients, and B to be a double-precision matrix with shape `(num_equations, nrhs)`, representing the ordinates.
    X is a matrix of unknowns with shape `(num_equations, nrhs)`.

    DSYSV will overwrite the matrices A and B. When terminated successfully, B will contain the solution to the system
    of linear equations. To retain their state, this function will copy the given arrays before invoking DSYSV.

    Furthermore, DSYSV assumes the matrix of coefficients A to be symmetrical, i.e., it will only use the upper-right
    triangle of A, whereas the remaining elements are ignored. For reasons of space efficiency, this function expects
    the coefficients to be given as an array with shape `(num_equations * (num_equations + 1)) // 2`, representing the
    elements of the upper-right triangle of A, where the rows are appended to each other and unspecified elements are
    omitted. This function will implicitly convert the given array into a matrix that is suited for DSYSV.

    :param coefficients:    An array of dtype `float64`, shape `(num_equations * (num_equations + 1)) // 2)`,
                            representing coefficients
    :param ordinates:       An array of dtype `float64`, shape `(num_equations)`, representing the ordinates
    :return:                An array of dtype `float64`, shape `(num_equations)`, representing the solution to the
                            system of linear equations
    """
    cdef float64[::1] result
    cdef intp r, c, i
    # The number of linear equations
    cdef int n = ordinates.shape[0]
    # Create the array A by copying the array `coefficients`. DSYSV requires the array A to be Fortran-contiguous...
    cdef float64[::1, :] a = matrix_float64(n, n)
    i = 0

    for r in range(n):
        for c in range(r, n):
            a[r, c] = coefficients[i]
            i += 1

    # Create the array B by copying the array `ordinates`. It will be overwritten with the solution to the system of
    # linear equations. DSYSV requires the array B to be Fortran-contiguous...
    cdef float64[::1, :] b = matrix_float64(n, 1)

    for r in range(n):
        b[r, 0] = ordinates[r]

    # 'U' if the upper-right triangle of A should be used, 'L' if the lower-left triangle should be used
    cdef char* uplo = 'U'
    # The number of right-hand sides, i.e, the number of columns of the matrix B
    cdef int nrhs = b.shape[1]
    # Variable to hold the result of the solver. Will be 0 when terminated successfully, unlike 0 otherwise
    cdef int info
    # We must query optimal value for the argument `lwork` (the length of the working array `work`)...
    cdef double worksize
    cdef int lwork = -1  # -1 means that the optimal value should be queried
    dsysv(uplo, &n, &nrhs, &a[0, 0], &n, <int*>0, &b[0, 0], &n, &worksize, &lwork, &info)  # Queries the optimal value
    lwork = <int>worksize
    # Allocate the working array...
    cdef double* work = <double*>malloc(lwork * sizeof(double))
    # Allocate another working array...
    cdef int* ipiv = <int*>malloc(n * sizeof(int))

    try:
        # Run the DSYSV solver...
        dsysv(uplo, &n, &nrhs, &a[0, 0], &n, ipiv, &b[0, 0], &n, work, &lwork, &info)

        if info == 0:
            # The solution has been computed successfully...
            result = b[:, 0]
            return result
        else:
            # An error occurred...
            raise ArithmeticError('DSYSV terminated with non-zero info code')
    finally:
        # Free the allocated memory...
        free(<void*>ipiv)
        free(<void*>work)
