/**
 * Implements classes that allow to find the best refinement of rules.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"
#include "tuples.h"
#include "predictions.h"
#include "rules.h"


/**
 * A struct that stores information about a potential refinement of a rule.
 */
struct Refinement {
    PredictionCandidate* head;
    uint32 featureIndex;
    float32 threshold;
    Comparator comparator;
    bool covered;
    uint32 coveredWeights;
    intp start;
    intp end;
    intp previous;
    IndexedFloat32Array* indexedArray;
    IndexedFloat32ArrayWrapper* indexedArrayWrapper;
};
