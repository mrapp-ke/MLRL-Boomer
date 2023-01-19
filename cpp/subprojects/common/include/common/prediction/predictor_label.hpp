/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/prediction_matrix_dense.hpp"
#include "common/prediction/prediction_matrix_sparse_binary.hpp"
#include "common/prediction/predictor.hpp"


/**
 * Defines an interface for all classes that allow to predict labels for given query examples.
 */
typedef IPredictor<DensePredictionMatrix<uint8>> ILabelPredictor;

/**
 * Defines an interface for all classes that allow to create instances of the type `ILabelPredictor`.
 */
typedef IPredictorFactory<DensePredictionMatrix<uint8>> ILabelPredictorFactory;

/**
 * Defines an interface for all classes that allow to predict sparse labels for given query examples.
 */
typedef IPredictor<BinarySparsePredictionMatrix> ISparseLabelPredictor;

/**
 * Defines an interface for all classes that allow to create instances of the type `ISparseLabelPredictor`.
 */
typedef IPredictorFactory<BinarySparsePredictionMatrix> ISparseLabelPredictorFactory;

/**
 * Defines an interface for all classes that allow to configure an `ILabelPredictor` or `ISparseLabelPredictor`.
 */
class ILabelPredictorConfig : public IPredictorConfig<DensePredictionMatrix<uint8>> {

    public:

        virtual ~ILabelPredictorConfig() override { };

        /**
         * Creates and returns a new object of type `ISparseLabelPredictorFactory` according to the specified
         * configuration.
         *
         * @param featureMatrix A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise access
         *                      to the feature values of the query examples to predict for
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `ISparseLabelPredictorFactory` that has been
         *                      created
         */
        virtual std::unique_ptr<ISparseLabelPredictorFactory> createSparsePredictorFactory(
            const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const = 0;

};
