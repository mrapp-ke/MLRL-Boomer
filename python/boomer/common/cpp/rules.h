/**
 * Provides classes that are used to build rule-based models.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "arrays.h"


/**
 * An enum that specifies all possible types of operators used by a condition of a rule.
 */
enum Comparator : uint32 {
    LEQ = 0,
    GR = 1,
    EQ = 2,
    NEQ = 3
};

/**
 * Stores information about a condition of a rule. It consists of the index of the feature, the condition corresponds
 * to, the type of the operator that is used by the condition, as well as a threshold. In addition, it stores the sum of
 * the weights of all elements, e.g. examples or bins, that are covered by the condition.
 */
struct Condition {
    uint32 featureIndex;
    Comparator comparator;
    float32 threshold;
    uint32 coveredWeights;
};
