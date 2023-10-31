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
struct Allocator : public View {
        /**
         * @param numElements   The number of elements in the view
         * @param init          True, if all elements in the view should be value-initialized, false otherwise
         */
        Allocator(uint32 numElements, bool init = false)
            : View(allocateMemory<typename View::value_type>(numElements, init), numElements) {}

        /**
         * @param other A reference to an object of type `Allocator` that should be moved
         */
        Allocator(Allocator<View>&& other) : View(std::move(other)) {
            other.array = nullptr;
        }

        virtual ~Allocator() override {
            freeMemory(View::array);
        }
};

/**
 * Allocates the memory, a `View` provides access to
 *
 * @tparam T The type of the values stored in the `View`
 */
template<typename T>
using AllocatedView = Allocator<View<T>>;

/**
 * Allocates the memory, a `Vector` provides access to
 *
 * @tparam T The type of the values stored in the `Vector`
 */
template<typename T>
using AllocatedVector = Allocator<Vector<T>>;

/**
 * An array that provides random read and write access to newly allocated memory.
 *
 * @tparam T The type of the values stored in the array
 */
template<typename T>
class Array : public WriteAccessibleViewDecorator<ReadAccessibleViewDecorator<VectorDecorator<AllocatedView<T>>>> {
    public:

        /**
         * @param numElements   The number of elements in the array
         * @param init          True, if all elements in the array should be value-initialized, false otherwise
         */
        Array(uint32 numElements, bool init = false)
            : WriteAccessibleViewDecorator<ReadAccessibleViewDecorator<VectorDecorator<AllocatedView<T>>>>(
              AllocatedView<T>(numElements, init)) {}
};
