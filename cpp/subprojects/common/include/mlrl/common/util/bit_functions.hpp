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
    static inline constexpr std::size_t bits() {
        return CHAR_BIT * sizeof(T);
    }

    /**
     * Returns the number of elements needed by an array of a specific type for storing a given number of bits.
     *
     * @tparam T        The type of the array
     * @param numBits   The number of bits to be stored
     * @return          The number of elements needed
     */
    template<typename T>
    static inline constexpr std::size_t bitArraySize(uint32 numBits) {
        return (numBits + bits<T>() - 1) / bits<T>();
    }

    /**
     * Returns the index of the element in an array of a specific type that stores the bit at a specific position,
     * starting from the beginning of the array.
     *
     * @tparam T            The type of the array
     * @param arrayIndex    The index of the bit
     * @return              The index of the element that stores the bit
     */
    template<typename T>
    static inline constexpr uint32 bitArrayOffset(uint32 arrayIndex) {
        return arrayIndex / bits<T>();
    }

    /**
     * Returns a bit mask that can be compared via bit-wise operators to a single element in an array of a specific
     * type. The mask consists exclusively of zeros, except for a single bit at a specific index that is set to one.
     *
     * @tparam T        The type of the array
     * @param bitIndex  The index of the bit to be set to one
     * @return          A value of template type `T` that stores the bit mask
     */
    template<typename T>
    static inline constexpr T bitArrayMask(uint32 bitIndex) {
        return 1U << (bitIndex % bits<T>());
    }

}
