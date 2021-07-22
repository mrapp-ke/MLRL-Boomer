/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include <stdexcept>


/**
 * Throws an exception if a given value is not greater than a specific threshold.
 *
 * @tparam T        The type of the value and threshold
 * @param value     The value
 * @param threshold The threshold
 */
template<typename T>
static inline constexpr void assertGreater(const T value, const T threshold) {
    if (value <= threshold) {
        throw std::invalid_argument("Value must be greater than " + std::to_string(threshold) + ", but is "
                                    + std::to_string(value));
    }
}

/**
 * Throws an exception if a given value not greater than or equal to a specific threshold.
 *
 * @tparam T        The type of the value and threshold
 * @param value     The value
 * @param threshold The threshold
 */
template<typename T>
static inline constexpr void assertGreaterOrEqual(const T value, const T threshold) {
    if (value < threshold) {
        throw std::invalid_argument("Value must be greater or equal to " + std::to_string(threshold) + ", but is "
                                    + std::to_string(value));
    }
}

/**
 * Throws an exception if a given value is not smaller than a specific threshold.
 *
 * @tparam T        The type of the value and threshold
 * @param value     The value
 * @param threshold The threshold
 */
template<typename T>
static inline constexpr void assertSmaller(const T value, const T threshold) {
    if (value >= threshold) {
        throw std::invalid_argument("Value must be smaller than " + std::to_string(threshold) + ", but is "
                                    + std::to_string(value));
    }
}

/**
 * Throws an exception if a given value is not smaller than or equal to a specific threshold.
 *
 * @tparam T        The type of the value and threshold
 * @param value     The value
 * @param threshold The threshold
 */
template<typename T>
static inline constexpr void assertSmallerOrEqual(const T value, const T threshold) {
    if (value > threshold) {
        throw std::invalid_argument("Value must be smaller or equal to " + std::to_string(threshold) + ", but is "
                                    + std::to_string(value));
    }
}

/**
 * Throws an exception if a given value is not a multiple of another value.
 *
 * @tparam T    The type of the values
 * @param value The value that should be a multiple of `other`
 * @param other The other value
 */
template<typename T>
static inline constexpr void assertMultiple(const T value, const T other) {
    if (value % other != 0) {
        throw std::invalid_argument("Value must be a multiple of " + std::to_string(other) + ", but is "
                                    + std::to_string(value));
    }
}
