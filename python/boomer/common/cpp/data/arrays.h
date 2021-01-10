/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include <algorithm>


/**
 * Sets all elements in an array to zero.
 *
 * @tparam T            The type of the array
 * @param a             A pointer to an array of template type `T`
 * @param numElements   The number of elements in the array
 */
template<typename T>
static inline void setArrayToZeros(T* a, uint32 numElements) {
    std::fill(a, a + numElements, 0);
}
