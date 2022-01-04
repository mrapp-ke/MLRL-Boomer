/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_matrix_c_contiguous.hpp"
#include "common/input/feature_matrix_csr.hpp"
#include "common/output/prediction_matrix_dense.hpp"
#include <memory>


/**
 * Defines an interface for all classes that allow to make predictions for given query examples using an existing
 * rule-based model and write them into a dense matrix.
 *
 * @tparam T The type of the values that are stored by the prediction matrix
 */
template<typename T>
class IPredictor {

    public:

        virtual ~IPredictor() { };

        /**
         * Obtains dense predictions for all examples in a C-contiguous matrix, using a specific rule-based model.
         *
         * @param featureMatrix     A reference to an object of type `CContiguousFeatureMatrix` that stores the feature
         *                          values of the examples
         * @param numLabels         The number of labels to predict for
         * @return                  An unique pointer to an object of type `DensePredictionMatrix` that stores the
         *                          predictions
         */
        virtual std::unique_ptr<DensePredictionMatrix<T>> predict(const CContiguousFeatureMatrix& featureMatrix,
                                                                  uint32 numLabels) const = 0;

        /**
         * Obtains dense predictions for all examples in a sparse CSR matrix, using a specific rule-based model.
         *
         * @param featureMatrix     A reference to an object of type `CsrFeatureMatrix` that stores the feature values
         *                          of the examples
         * @param numLabels         The number of labels to predict for
         * @return                  An unique pointer to an object of type `DensePredictionMatrix` that stores the
         *                          predictions
         */
        virtual std::unique_ptr<DensePredictionMatrix<T>> predict(const CsrFeatureMatrix& featureMatrix,
                                                                  uint32 numLabels) const = 0;

};
