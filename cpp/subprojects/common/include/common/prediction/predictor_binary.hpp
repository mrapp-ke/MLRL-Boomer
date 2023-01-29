/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/prediction_matrix_dense.hpp"
#include "common/prediction/prediction_matrix_sparse_binary.hpp"
#include "common/prediction/predictor.hpp"

/**
 * Defines an interface for all classes that allow to predict binary labels for given query examples.
 */
class IBinaryPredictor : public IPredictor<DensePredictionMatrix<uint8>> {
    public:

        virtual ~IBinaryPredictor() override {};
};

/**
 * Defines an interface for all classes that allow to create instances of the type `IBinaryPredictor`.
 */
class IBinaryPredictorFactory : public IPredictorFactory<IBinaryPredictor> {
    public:

        virtual ~IBinaryPredictorFactory() override {};
};

/**
 * Defines an interface for all classes that allow to predict sparse binary labels for given query examples.
 */
class ISparseBinaryPredictor : public IPredictor<BinarySparsePredictionMatrix> {
    public:

        virtual ~ISparseBinaryPredictor() override {};
};

/**
 * Defines an interface for all classes that allow to create instances of the type `ISparseBinaryPredictor`.
 */
class ISparseBinaryPredictorFactory : public IPredictorFactory<ISparseBinaryPredictor> {
    public:

        virtual ~ISparseBinaryPredictorFactory() override {};
};

/**
 * Defines an interface for all classes that allow to configure an `IBinaryPredictor` or `ISparseBinaryPredictor`.
 */
class IBinaryPredictorConfig : public IPredictorConfig<IBinaryPredictorFactory> {
    public:

        virtual ~IBinaryPredictorConfig() override {};

        /**
         * Creates and returns a new object of type `ISparseBinaryPredictorFactory` according to the specified
         * configuration.
         *
         * @param featureMatrix A reference to an object of type `IRowWiseFeatureMatrix` that provides row-wise access
         *                      to the feature values of the query examples to predict for
         * @param numLabels     The number of labels to predict for
         * @return              An unique pointer to an object of type `ISparseBinaryPredictorFactory` that has been
         *                      created
         */
        virtual std::unique_ptr<ISparseBinaryPredictorFactory> createSparsePredictorFactory(
          const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const = 0;
};
