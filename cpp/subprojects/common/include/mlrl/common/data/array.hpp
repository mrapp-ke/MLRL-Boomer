/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view.hpp"

/**
 * An array that provides random read and write access to newly allocated memory.
 *
 * @tparam T The type of the values stored in the array
 */
template<typename T>
class Array : public IndexableViewDecorator<ViewDecorator<AllocatedView<T>>> {
    public:

        /**
         * @param numElements   The number of elements in the array
         * @param init          True, if all elements in the array should be value-initialized, false otherwise
         */
        Array(uint32 numElements, bool init = false)
            : IndexableViewDecorator<ViewDecorator<AllocatedView<T>>>(AllocatedView<T>(numElements, init)) {}
};
