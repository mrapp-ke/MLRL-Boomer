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
class DenseVector : public ResizableVectorDecorator<WritableVectorDecorator<ResizableVector<T>>> {
    public:

        /**
         * @param numElements   The number of elements in the vector
         * @param init          True, if all elements in the vector should be value-initialized, false otherwise
         */
        DenseVector(uint32 numElements, bool init = false)
            : ResizableVectorDecorator<WritableVectorDecorator<ResizableVector<T>>>(
              ResizableVector<T>(numElements, init)) {}

        virtual ~DenseVector() override {};
};
