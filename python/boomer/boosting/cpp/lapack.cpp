#include "lapack.h"
#include <stdlib.h>

dsysv_t dsysvFunction;


float64* dsysv(float64* coefficients, float64* invertedOrdinates, int n, float64 l2RegularizationWeight) {
    // Create the array A by copying the array `coefficients`. DSYSV requires the array A to be Fortran-contiguous...
    float64* a = (float64*) malloc(n * n * sizeof(float64));
    int i = 0;

    for (int c = 0; c < n; c++) {
        int offset = c * n;

        for (int r = 0; r < c + 1; r++) {
            float64 tmp = coefficients[i];

            if (r == c) {
                tmp += l2RegularizationWeight;
            }

            a[offset + r] = tmp;
            i++;
        }
    }

    // Create the array B by copying the array `invertedOrdinates` and inverting its elements. It will be overwritten
    // with the solution to the system of linear equations. DSYSV requires the array B to be Fortran-contiguous...
    float64* b = (float64*) malloc(n * sizeof(float64));

    for (int i = 0; i < n; i++) {
        b[i] = -invertedOrdinates[i];
    }

    // "U" if the upper-right triangle of A should be used, "L" if the lower-left triangle should be used
    char* uplo = "U";
    // The number of right-hand sides, i.e, the number of columns of the matrix B
    int nrhs = 1;
    // Variable to hold the result of the solver. Will be 0 when terminated successfully, unlike 0 otherwise
    int info;
    // We must query the optimal value for the argument `lwork` (the length of the working array `work`)...
    double worksize;
    int lwork = -1;  // -1 means that the optimal value should be queried
    dsysvFunction(uplo, &n, &nrhs, a, &n, (int*) 0, b, &n, &worksize, &lwork, &info);
    lwork = (int) worksize;
    // Allocate working arrays...
    double* work = (double*) malloc(lwork * sizeof(double));
    int* ipiv = (int*) malloc(n * sizeof(int));
    // Run the DSYSV solver...
    dsysvFunction(uplo, &n, &nrhs, a, &n, ipiv, b, &n, work, &lwork, &info);
    // Free the allocated memory...
    free(a);
    free(ipiv);
    free(work);

    if (info == 0) {
        // The solution has been computed successfully...
        return b;
    } else {
        // TODO An error occurred...
        return b;
    }
}
