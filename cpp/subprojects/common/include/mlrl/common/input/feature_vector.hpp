/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/indexed_value.hpp"
#include "mlrl/common/data/view_vector.hpp"
#include "mlrl/common/input/missing_feature_vector.hpp"

/**
 * An one-dimensional sparse vector that stores the values of training examples for a certain feature, as well as the
 * indices of examples with missing feature values.
 */
class FeatureVector final
    : public ResizableVectorDecorator<WritableVectorDecorator<ResizableVector<IndexedValue<float32>>>>,
      public MissingFeatureVector {
    public:

        /**
         * @param numElements The number of elements in the vector
         */
        FeatureVector(uint32 numElements);

        /**
         * Sorts the elements in the vector in ascending order based on their values.
         */
        void sortByValues();
};
