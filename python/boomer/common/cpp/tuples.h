/**
 * Provides type definitions of commonly used tuples.
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
 *  A struct that stores information about the examples that are contained by a bin.
 */
struct Bin {
    Bin() : numExamples(0), minValue(std::numeric_limits<float32>::max()),
            maxValue(std::numeric_limits<float32>::min()) { };
    uint32 numExamples;
    float32 minValue;
    float32 maxValue;
};
