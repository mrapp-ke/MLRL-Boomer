/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector.hpp"

/**
 * A vector that provides random read and write access, as well as read and write access via iterators, to the values
 * stored in a newly allocated array.
 *
 * @tparam T The type of the values stored in the vector
 */
template<typename T>
class DenseVector final : public IterableVectorDecorator<VectorDecorator<AllocatedVector<T>>> {
    public:

        /**
         * @param numElements   The number of elements in the vector
         * @param init          True, if all elements in the vector should be value-initialized, false otherwise
         */
        DenseVector(uint32 numElements, bool init = false)
            : IterableVectorDecorator<VectorDecorator<AllocatedVector<T>>>(AllocatedVector<T>(numElements, init)) {}

        /**
         * @param other A reference to an object of type `AllocatedVector` that should be moved
         */
        DenseVector(AllocatedVector<T>&& other)
            : IterableVectorDecorator<VectorDecorator<AllocatedVector<T>>>(AllocatedVector<T>(std::move(other))) {}
};

/**
 * A vector that provides random read and write access, as well as read and write access via iterators, to the values
 * stored in a newly allocated array, which can be resized.
 *
 * @tparam T The type of the values stored in the vector
 */
template<typename T>
class ResizableDenseVector final
    : public ResizableVectorDecorator<IterableVectorDecorator<VectorDecorator<ResizableVector<T>>>> {
    public:

        /**
         * @param numElements   The number of elements in the vector
         * @param init          True, if all elements in the vector should be value-initialized, false otherwise
         */
        ResizableDenseVector(uint32 numElements, bool init = false)
            : ResizableVectorDecorator<IterableVectorDecorator<VectorDecorator<ResizableVector<T>>>>(
              ResizableVector<T>(numElements, init)) {}
};
