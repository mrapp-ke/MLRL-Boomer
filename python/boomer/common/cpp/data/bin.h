/**
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 */
#pragma once

#include "types.h"
#include <limits>


/**
 *  A struct that stores information about the examples that are contained by a bin.
 */
struct Bin {
    Bin() : sumOfWeights(0), minValue(std::numeric_limits<float32>::infinity()),
            maxValue(-std::numeric_limits<float32>::infinity()) { };
    uint32 index;
    uint32 sumOfWeights;
    float32 minValue;
    float32 maxValue;
};
