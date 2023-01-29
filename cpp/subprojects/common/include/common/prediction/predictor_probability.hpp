/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/prediction_matrix_dense.hpp"
#include "common/prediction/predictor.hpp"

/**
 * Defines an interface for all classes that allow to predict probability estimates for given query examples.
 */
class IProbabilityPredictor : public IPredictor<DensePredictionMatrix<float64>> {
    public:

        virtual ~IProbabilityPredictor() override {};
};

/**
 * Defines an interface for all classes that allow to create instances of the type `IProbabilityPredictor`.
 */
class IProbabilityPredictorFactory : public IPredictorFactory<IProbabilityPredictor> {
    public:

        virtual ~IProbabilityPredictorFactory() override {};
};

/**
 * Defines an interface for all classes that allow to configure a `IProbabilityPredictor`.
 */
class IProbabilityPredictorConfig : public IPredictorConfig<IProbabilityPredictorFactory> {
    public:

        virtual ~IProbabilityPredictorConfig() override {};
};
