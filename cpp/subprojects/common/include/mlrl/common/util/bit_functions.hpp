/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/types.hpp"

#include <climits>

namespace util {

    /**
     * Returns the number of bits used by a certain type.
     *
     * @tparam T    The type
     * @return      The number of bits
     */
    template<typename T>
    static inline constexpr uint32 bits() {
        return static_cast<uint32>(CHAR_BIT * sizeof(T));
    }

    /**
     * Returns the number of elements needed by an array of a specific type for storing a given number of integers, each
     * with a specific number of bits.
     *
     * @tparam T                The type of the array
     * @param numIntegers       The number of integers to be stored
     * @param numBitsPerInteger The number of bits per integer
     * @return                  The number of elements needed
     */
    template<typename T>
    static inline constexpr uint32 getBitArraySize(uint32 numIntegers, uint32 numBitsPerInteger) {
        uint32 numIntegersPerElement = bits<T>() / numBitsPerInteger;
        return numIntegers / numIntegersPerElement + (numIntegers % numIntegersPerElement != 0);
    }

    /**
     * Returns the number of values representable by a given number of bits.
     *
     * @param numBits   The number of bits
     * @return          The number of representable values
     */
    static inline uint32 getNumBitCombinations(uint32 numBits) {
        return static_cast<uint32>(std::pow(2, numBits));
    }
}
