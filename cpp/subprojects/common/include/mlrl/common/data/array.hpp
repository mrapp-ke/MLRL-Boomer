/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector.hpp"
#include "mlrl/common/util/memory.hpp"

/**
 * Allocates the memory, a view provides access to.
 *
 * @tparam Base The type of the view
 */
template<typename View>
struct AllocatedView : public View {
        /**
         * @param numElements The number of elements in the view
         */
        AllocatedView(uint32 numElements) : AllocatedView(numElements, false) {}

        /**
         * @param numElements   The number of elements in the view
         * @param init          True, if all elements in the view should be value-initialized, false otherwise
         */
        AllocatedView(uint32 numElements, bool init)
            : View(allocateMemory<typename View::value_type>(numElements, init), numElements) {}

        /**
         * @param other A reference to an object of type `AllocatedView` that should be moved
         */
        AllocatedView(AllocatedView<View>&& other) : View(std::move(other)) {
            other.array = nullptr;
        }

        virtual ~AllocatedView() override {
            freeMemory(View::array);
        }
};

/**
 * An array that provides random read and write access to newly allocated memory.
 *
 * @tparam T The type of the values stored in the array
 */
template<typename T>
class Array : public AccessibleVectorDecorator<AllocatedView<View<T>>> {
    public:

        /**
         * @param numElements The number of elements in the array
         */
        Array(uint32 numElements)
            : AccessibleVectorDecorator<AllocatedView<View<T>>>(AllocatedView<View<T>>(numElements)) {}

        /**
         * @param numElements   The number of elements in the array
         * @param init          True, if all elements in the array should be value-initialized, false otherwise
         */
        Array(uint32 numElements, bool init)
            : AccessibleVectorDecorator<AllocatedView<View<T>>>(AllocatedView<View<T>>(numElements, init)) {}
};
