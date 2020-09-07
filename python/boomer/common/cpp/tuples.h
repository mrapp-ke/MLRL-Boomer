/**
 * Provides type definitions of tuples, as well as corresponding utility functions.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"


/**
 * A struct that stores a value of type float32 and a corresponding index that refers to the (original) position of the
 * value in an array.
 */
struct IndexedFloat32 {
    intp index;
    float32 value;
};

/**
 * A struct that contains a pointer to a C-array of type `IndexedFloat32`. The attribute `num_elements` specifies how
 * many elements the array contains.
 */
struct IndexedFloat32Array {
    IndexedFloat32* data;
    intp numElements;
};

/**
 * A struct that stores a value of type float64 and a corresponding index that refers to the (original) position of the
 * value in an array.
 */
struct IndexedFloat64 {
    intp index;
    float64 value;
};
