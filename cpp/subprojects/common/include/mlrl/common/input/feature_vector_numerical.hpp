/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/indexed_value.hpp"
#include "mlrl/common/data/view_vector.hpp"

/**
 * A feature vector that stores the values of training examples for a certain numerical feature, except for the examples
 * associated with a sparse value.
 */
class MLRLCOMMON_API NumericalFeatureVector : public Vector<IndexedValue<float32>> {
    public:

        /**
         * The value of sparse elements not explicitly stored in the vector.
         */
        float32 sparseValue;

        /**
         * True, if there are any sparse elements not explicitly stored in the vector, false otherwise.
         */
        bool sparse;

        /**
         * @param array         A pointer to an array of type `IndexedValue<float32>` that stores the values in the
         *                      feature vector
         * @param numElements   The number of elements in the vector, excluding those associated with the sparse value
         */
        NumericalFeatureVector(IndexedValue<float32>* array, uint32 numElements);

        /**
         * @param other A const reference to an object of type `NumericalFeatureVector` that should be copied
         */
        NumericalFeatureVector(const NumericalFeatureVector& other);

        /**
         * @param other A reference to an object of type `NumericalFeatureVector` that should be moved
         */
        NumericalFeatureVector(NumericalFeatureVector&& other);

        virtual ~NumericalFeatureVector() override {};
};
