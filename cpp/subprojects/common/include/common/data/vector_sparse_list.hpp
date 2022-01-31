/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/indexed_value.hpp"
#include <forward_list>


/**
 * An one-dimensional sparse vector that stores elements, consisting of an index and a value, in a linked list.
 *
 * @tparam T The type of the values that are stored in the vector
 */
template<typename T>
using SparseListVector = std::forward_list<IndexedValue<T>>;
