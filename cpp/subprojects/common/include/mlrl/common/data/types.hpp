/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include <cmath>
#include <limits>

typedef int int32;
typedef long int int64;
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef float float32;
typedef double float64;

/**
 * Returns whether two values `a` and `b` are equal or not.
 *
 * @tparam T    The type of the values to be compared
 * @param a     The first value
 * @param b     The second value
 * @return      True if the given values are equal, false otherwise
 */
template<typename T>
inline bool isEqual(T a, T b) {
    return a == b;
}

/**
 * Returns whether two values `a` and `b` of type `float32` are (approximately) equal or not.
 *
 * @param a The first value
 * @param b The second value
 * @return  True if the given values are equal, false otherwise
 */
template<>
inline bool isEqual(float32 a, float32 b) {
    return std::fabs(a - b)
           <= std::numeric_limits<float32>::epsilon() * std::fmax(1, std::fmax(std::fabs(a), std::fabs(b)));
}

/**
 * Returns whether two values `a` and `b` of type `float64` are (approximately) equal or not.
 *
 * @param a The first value
 * @param b The second value
 * @return  True if the given values are equal, false otherwise
 */
template<>
inline bool isEqual(float64 a, float64 b) {
    return std::fabs(a - b)
           <= std::numeric_limits<float64>::epsilon() * std::fmax(1, std::fmax(std::fabs(a), std::fabs(b)));
}

/**
 * Returns whether a specific value is equal to zero or not.
 *
 * @param a The value
 * @return  True, if the given value is equal to zero, false otherwise
 */
template<typename T>
inline bool isEqualToZero(T a) {
    return isEqual(a, (T) 0);
}
