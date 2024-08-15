/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view.hpp"

#include <utility>

/**
 * A view that provides random access to integer values, each with a specific number of bits, stored in a pre-allocated
 * array of a specific size.
 */
class BitView : public BaseView<uint32> {
    public:

        /**
         * The number of bits per element in the view.
         */
        uint32 numBitsPerElement;

        /**
         * @param array         A pointer to an array of type `uint32` that stores the values, the view should provide
         *                      access to
         * @param dimensions    The number of elements in each dimension of the view
         */
        BitView(uint32* array, std::initializer_list<uint32> dimensions)
            : BaseView<uint32>(array), numBitsPerElement(dimensions.begin()[1]) {}

        /**
         * @param array             A pointer to an array of type `uint32` that stores the values, the view should
         *                          provide access to
         * @param numElements       The number of elements in the view
         * @param numBitsPerElement The number of bits per element in the view
         */
        BitView(uint32* array, uint32 numElements, uint32 numBitsPerElement)
            : BaseView<uint32>(array), numBitsPerElement(numBitsPerElement) {}

        /**
         * @param other A const reference to an object of type `BitView` that should be copied
         */
        BitView(const BitView& other) : BaseView<uint32>(other), numBitsPerElement(other.numBitsPerElement) {}

        /**
         * @param other A reference to an object of type `BitView` that should be moved
         */
        BitView(BitView&& other) : BaseView<uint32>(std::move(other)), numBitsPerElement(other.numBitsPerElement) {}

        virtual ~BitView() override {}
};
