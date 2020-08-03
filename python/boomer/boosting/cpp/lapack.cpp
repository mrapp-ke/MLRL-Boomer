#include "lapack.h"
#include <stdlib.h>


Lapack::Lapack(dsysv_t dsysvFunction) {
    dsysvFunction_ = dsysvFunction;
}

void Lapack::dsysv(float64* coefficients, float64* invertedOrdinates, float64* tmpArray1, int* tmpArray2,
                   float64* output, int n, float64 l2RegularizationWeight) {
    // Copy the values in the arrays `invertedOrdinates` and `coefficients` to the arrays `output` and `tmpArray1`,
    // respectively...
    int i = 0;

    for (int c = 0; c < n; c++) {
        output[c] = -invertedOrdinates[c];
        int offset = c * n;

        for (int r = 0; r < c + 1; r++) {
            float64 tmp = coefficients[i];

            if (r == c) {
                tmp += l2RegularizationWeight;
            }

            tmpArray1[offset + r] = tmp;
            i++;
        }
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
    dsysvFunction_(uplo, &n, &nrhs, tmpArray1, &n, (int*) 0, output, &n, &worksize, &lwork, &info);
    lwork = (int) worksize;
    // Allocate working arrays...
    double* work = (double*) malloc(lwork * sizeof(double));
    // Run the DSYSV solver...
    dsysvFunction_(uplo, &n, &nrhs, tmpArray1, &n, tmpArray2, output, &n, work, &lwork, &info);
    // Free the allocated memory...
    free(work);

    if (info != 0) {
        // TODO An error occurred...
    }
}
