/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/types.hpp"


/**
 * A tuple that consists of two values of the same type.
 *
 * @tparam T The type of the values
 */
template<class T>
struct Tuple {

    Tuple() { };

    /**
     * @param f The first value
     * @param s The second value
     */
    Tuple(T f, T s) : first(f), second(s) { };

    /**
     * The first value.
     */
    T first;

    /**
     * The second value.
     */
    T second;

};
