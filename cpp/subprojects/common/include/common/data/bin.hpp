/**
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"
#include <limits>


/**
 *  A struct that stores information about the examples that are contained by a bin.
 */
struct Bin {
    Bin() : numExamples(0), minValue(std::numeric_limits<float32>::infinity()),
            maxValue(-std::numeric_limits<float32>::infinity()) { };
    uint32 numExamples; // TODO Currently this is only used to check if a bin is empty. Can be removed and the check can be based on minValue or maxValue instead.
    float32 minValue;
    float32 maxValue;
};
