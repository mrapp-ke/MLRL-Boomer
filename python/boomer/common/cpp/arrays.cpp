#include "arrays.h"


/**
 * Sets all elements in an one- or two-dimensional array to zero.
 *
 * @tparam T            The type of the array
 * @param a             A pointer to an array of template type `T`
 * @param numElements   The number of elements in the array
 */
template<typename T>
static inline void setToZeros(T* a, uint32 numElements) {
    for (uint32 i = 0; i < numElements; i++) {
        a[i] = 0;
    }
}
