/**
 * Provides type definitions of tuples, as well as corresponding utility functions.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"


/**
 * A tuple that consists of an index and a value.
 */
template<class T>
struct IndexedValue {
    uint32 index;
    T value;
};

/**
 * A struct that stores a value of type float32 and a corresponding index that refers to the (original) position of the
 * value in an array.
 */
struct IndexedFloat32 {
    uint32 index;
    float32 value;
};

/**
 * A struct that contains a pointer to an array of type `IndexedFloat32`. The attribute `numElements` specifies how many
 * elements the array contains.
 */
struct IndexedFloat32Array {
    IndexedFloat32* data;
    uint32 numElements;
};


namespace tuples {

    /**
     * Compares the values of two structs of type `IndexedValue`.
     *
     * @param a A pointer to the first struct
     * @param b A pointer to the second struct
     * @return  -1 if the value of the first struct is smaller than the value of the second struct, 0 if both values are
     *          equal, or 1 if the value of the first struct is greater than the value of the second struct
     */
    template<class T>
    static inline int compareIndexedValue(const void* a, const void* b) {
        T v1 = ((IndexedValue<T>*) a)->value;
        T v2 = ((IndexedValue<T>*) b)->value;
        return v1 < v2 ? -1 : (v1 == v2 ? 0 : 1);
    }

}
