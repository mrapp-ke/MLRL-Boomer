/**
 * Provides type definitions of tuples, as well as corresponding utility functions.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include <limits>


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

/**
 *  A struct that stores information about the examples that are contained by a bin.
 */
struct Bin {
    Bin() : numExamples(0), minValue(std::numeric_limits<float32>::max()),
            maxValue(std::numeric_limits<float32>::min()) { };
    uint32 numExamples;
    float32 minValue;
    float32 maxValue;
};
