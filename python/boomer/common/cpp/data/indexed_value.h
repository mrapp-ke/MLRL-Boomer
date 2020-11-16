/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "types.h"


/**
 * A tuple that consists of an index and a value.
 */
template<class T>
struct IndexedValue {
    uint32 index;
    T value;
};
