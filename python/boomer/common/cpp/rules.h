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
