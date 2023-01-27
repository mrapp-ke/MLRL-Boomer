/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/prediction_matrix_dense.hpp"
#include "common/prediction/predictor.hpp"

/**
 * Defines an interface for all classes that allow to predict regression scores for given query examples.
 */
class IScorePredictor : public IPredictor<DensePredictionMatrix<float64>> {
    public:

        virtual ~IScorePredictor() override { };

};

/**
 * Defines an interface for all classes that allow to create instances of the type `IScorePredictor`.
 */
class IScorePredictorFactory : public IPredictorFactory<IScorePredictor> {
    public:

        virtual ~IScorePredictorFactory() override { };

};

/**
 * Defines an interface for all classes that allow to configure a `IScorePredictor`.
 */
class IScorePredictorConfig : public IPredictorConfig<IScorePredictorFactory> {
    public:

        virtual ~IScorePredictorConfig() override { };

};
