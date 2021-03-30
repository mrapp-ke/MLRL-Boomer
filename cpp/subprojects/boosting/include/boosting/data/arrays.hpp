/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include <algorithm>


namespace boosting {

    /**
     * Sets all elements in an array to a specific value.
     *
     * @tparam T            The type of the array
     * @param a             A pointer to an array of template type `T`
     * @param numElements   The number of elements in the array
     * @param value         The value to be set
     */
    template<typename T>
    static inline void setArrayToValue(T* a, uint32 numElements, T value) {
        std::fill(a, a + numElements, value);
    }

}
