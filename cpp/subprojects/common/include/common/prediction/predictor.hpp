/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/view_c_contiguous.hpp"
#include "common/data/view_csr.hpp"
#include <memory>


/**
 * Defines an interface for all classes that allow to make prediction for given query examples.
 *
 * @tparam PredictionMatrix The type of the matrix that is used to store the predictions
 */
template<typename PredictionMatrix>
class IPredictor {

    public:

        virtual ~IPredictor() { };

        /**
         * Obtains and returns predictions for all query examples in a C-contiguous matrix.
         *
         * @param featureMatrix A reference to an object of type `CContiguousConstView` that stores the feature values
         *                      of the query examples
         * @return              An unique pointer to an object of template type `PredictionMatrix` that stores the
         *                      predictions
         */
        virtual std::unique_ptr<PredictionMatrix> predict(
            const CContiguousConstView<const float32>& featureMatrix) const = 0;

        /**
         * Obtains and returns predictions for all query examples in a sparse CSR matrix.
         *
         * @param featureMatrix A reference to an object of type `CsrConstView` that stores the feature values of the
         *                      query examples
         * @return              An unique pointer to an object of template type `PredictionMatrix` that stores the
         *                      predictions
         */
        virtual std::unique_ptr<PredictionMatrix> predict(const CsrConstView<const float32>& featureMatrix) const = 0;

};
