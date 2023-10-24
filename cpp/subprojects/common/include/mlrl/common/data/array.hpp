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
