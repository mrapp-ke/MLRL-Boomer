/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"


/**
 * A tuple that consists of an index and a value.
 *
 * @tparam T The type of the value
 */
template<class T>
struct IndexedValue {

    IndexedValue() { };

    /**
     * @param i The index
     * @param v The value
     */
    IndexedValue(uint32 i, T v) : index(i), value(v) { };

    /**
     * The index.
     */
    uint32 index;

    /**
     * The value.
     */
    T value;

};
