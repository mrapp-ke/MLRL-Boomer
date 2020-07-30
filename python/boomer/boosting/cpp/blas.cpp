#include "blas.h"

ddot_t ddotFunction;


float64 ddot(float64* x, float64* y, int n) {
    // Storage spacing between the elements of the arrays x and y
    int inc = 1;
    // Invoke the DDOT routine...
    return ddotFunction(&n, x, &inc, y, &inc);
}
