/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector.hpp"

/**
 * Allows to resize a vector.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
class ResizableVectorDecorator : public Vector {
    private:

        uint32 maxCapacity_;

    public:

        /**
         * @param view The view, the vector should be backed by
         */
        ResizableVectorDecorator(typename Vector::view_type&& view)
            : Vector(std::move(view)), maxCapacity_(view.numElements) {}

        virtual ~ResizableVectorDecorator() override {};

        /**
         * Sets the number of elements in the vector.
         *
         * @param numElements   The number of elements to be set
         * @param freeMemory    True, if unused memory should be freed, if possible, false otherwise
         */
        void setNumElements(uint32 numElements, bool freeMemory) {
            if (numElements < maxCapacity_) {
                if (freeMemory) {
                    Vector::view_.array = reallocateMemory(Vector::view_.array, numElements);
                    maxCapacity_ = numElements;
                }
            } else if (numElements > maxCapacity_) {
                Vector::view_.array = reallocateMemory(Vector::view_.array, numElements);
                maxCapacity_ = numElements;
            }

            Vector::view_.numElements = numElements;
        }
};

/**
 * An one-dimensional vector that provides random access to a fixed number of elements stored in a C-contiguous array.
 *
 * @tparam T The type of the data that is stored in the vector
 */
template<typename T>
class MLRLCOMMON_API DenseVector : public VectorView<T> {
    private:

        uint32 maxCapacity_;

    public:

        /**
         * @param numElements The number of elements in the vector
         */
        DenseVector(uint32 numElements);

        /**
         * @param numElements   The number of elements in the vector
         * @param init          True, if all elements in the vector should be value-initialized, false otherwise
         */
        DenseVector(uint32 numElements, bool init);

        virtual ~DenseVector() override;

        /**
         * Sets the number of elements in the vector.
         *
         * @param numElements   The number of elements to be set
         * @param freeMemory    True, if unused memory should be freed, if possible, false otherwise
         */
        void setNumElements(uint32 numElements, bool freeMemory);
};
