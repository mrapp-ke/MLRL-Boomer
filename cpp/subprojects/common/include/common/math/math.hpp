/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"


/**
 * Allows to compute the arithmetic mean of several floating point values `x_1, ..., x_n` in an iterative manner, which
 * prevents overflows.
 *
 * This function must be invoked for each value as follows:
 * `mean_1 = iterativeArithmeticMean(1, x_1, 0); ...; mean_n = iterativeArithmeticMean(n, x_n, mean_n-1)`
 *
 * @tparam T    The type of the values
 * @param n     The index of the value, starting at 1
 * @param x     The n-th value
 * @return      The arithmetic mean of all values provided so far
 */
template<class T>
static inline T iterativeArithmeticMean(uint32 n, T x, T mean) {
    return mean + ((x - mean) / (T) n);
}
