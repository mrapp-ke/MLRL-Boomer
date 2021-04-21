/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"


/**
 * Calculates and returns a hash value from an array of type `uint32`.
 *
 * @param a             A pointer to an array of type `uint32`
 * @param numElements   The number of elements in the array
 * @return              The hash value
 */
static inline std::size_t hashArray(const uint32* a, uint32 numElements) {
    std::size_t hashValue = (std::size_t) numElements;

    for (uint32 i = 0; i < numElements; i++) {
        hashValue ^= a[i] + 0x9e3779b9 + (hashValue << 6) + (hashValue >> 2);
    }

    return hashValue;
}

/**
 * Returns whether two arrays of type `uint32` are equal or not.
 *
 * @param first             A pointer to an array of type `uint32`
 * @param firstNumElements  The number of elements in the array `first`
 * @param second            A pointer to another array of type `uint32`
 * @param secondNumElements The number of elements in the array `second`
 * @return                  True, if both arrays are equal, false otherwise
 */
static inline bool compareArrays(const uint32* first, uint32 firstNumElements, const uint32* second,
                                 uint32 secondNumElements) {
    if (firstNumElements != secondNumElements) {
        return false;
    }

    for (uint32 i = 0; i < firstNumElements; i++) {
        if (first[i] != second[i]) {
            return false;
        }
    }

    return true;
}
