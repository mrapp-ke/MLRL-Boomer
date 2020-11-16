/**
 * Provides commonly used functions that implement mathematical operations.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#include "../../common/cpp/data/types.h"


/**
 * Computes and returns the square of the L2 norm of a specific array, i.e. the sum of the squares of its elements. To
 * obtain the actual L2 norm, the square-root of the result provided by this function must be computed.
 *
 * @param a A pointer to an array of type `float64`, shape `(n)`
 * @param n The number of elements in the array `a`
 * @return  A scalar of type `float64`, representing the square of the L2 norm of the given array
*/
static inline float64 l2NormPow(const float64* a, uint32 n) {
    float64 result = 0;

    for (uint32 i = 0; i < n; i++) {
        float64 value = a[i];
        result += (value * value);
    }

    return result;
}
