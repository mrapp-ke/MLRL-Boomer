/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/prediction_matrix_dense.hpp"
#include "common/prediction/predictor.hpp"


/**
 * Defines an interface for all classes that allow to predict probability estimates for given query examples.
 */
typedef IPredictor<DensePredictionMatrix<float64>> IProbabilityPredictor;

/**
 * Defines an interface for all classes that allow to create instances of the type `IProbabilityPredictor`.
 */
typedef IPredictorFactory<DensePredictionMatrix<uint8>> IProbabilityPredictorFactory;
