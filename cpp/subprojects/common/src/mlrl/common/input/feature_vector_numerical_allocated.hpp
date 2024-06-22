/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_vector_numerical.hpp"

#include <utility>

/**
 * Allocates the memory, a `NumericalFeatureVector` provides access to.
 */
class MLRLCOMMON_API AllocatedNumericalFeatureVector : public ResizableAllocator<NumericalFeatureVector> {
    public:

        /**
         * @param numElements   The number of elements in the vector, excluding those associated with the sparse value
         * @param sparseValue   The value of sparse elements not explicitly stored in the vector
         * @param sparse        True, if there are any sparse elements not explicitly stored in the vector, false
         *                      otherwise
         */
        explicit AllocatedNumericalFeatureVector(uint32 numElements, float32 sparseValue = 0, bool sparse = false)
            : ResizableAllocator<NumericalFeatureVector>(numElements) {
            NumericalFeatureVector::sparseValue = sparseValue;
            NumericalFeatureVector::sparse = sparse;
        }

        /**
         * @param other A reference to an object of type `AllocatedNumericalFeatureVector` that should be copied
         */
        AllocatedNumericalFeatureVector(const AllocatedNumericalFeatureVector& other)
            : ResizableAllocator<NumericalFeatureVector>(other) {
            throw std::runtime_error("Objects of type AllocatedNumericalFeatureVector cannot be copied");
        }

        /**
         * @param other A reference to an object of type `AllocatedNumericalFeatureVector` that should be moved
         */
        AllocatedNumericalFeatureVector(AllocatedNumericalFeatureVector&& other)
            : ResizableAllocator<NumericalFeatureVector>(std::move(other)) {}

        virtual ~AllocatedNumericalFeatureVector() override {}
};
