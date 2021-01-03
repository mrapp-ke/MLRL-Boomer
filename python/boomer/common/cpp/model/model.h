/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../data/matrix_dense.h"
#include "../input/feature_matrix_c_contiguous.h"
#include "../input/feature_matrix_csr.h"


/**
 * Defines an interface for all classes that represent a model.
 */
class IModel {

    public:

        virtual ~IModel() { };

        /**
         * Adds the scores that are predicted by the model to a given matrix of predictions.
         *
         * @param featureMatrix     A reference to an object of type `CContinuousFeatureMatrix` that stores the feature
         *                          values of the examples
         * @param predictionMatrix  A reference to an object of type `DenseMatrix` that stores the predictions to be
         *                          updated
         */
        virtual void predict(const CContiguousFeatureMatrix& featureMatrix,
                             DenseMatrix<float64>& predictionMatrix) const = 0;

        /**
         * Adds the scores that are predicted by the model to a given matrix of predictions.
         *
         * @param featureMatrix     A reference to an object of type `CsrFeatureMatrix` that stores the feature values
         *                          of the examples
         * @param predictionMatrix  A reference to an object of type `DenseMatrix` that stores the predictions to be
         *                          updated
         */
        virtual void predict(const CsrFeatureMatrix& featureMatrix, DenseMatrix<float64>& predictionMatrix) const = 0;

};
