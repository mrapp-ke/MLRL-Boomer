/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/prediction_matrix_dense.hpp"
#include "common/prediction/predictor.hpp"


/**
 * Defines an interface for all classes that allow to predict regression scores for given query examples.
 */
typedef IPredictor<DensePredictionMatrix<float64>> IScorePredictor;

/**
 * Defines an interface for all classes that allow to create instances of the type `IScorePredictor`.
 */
typedef IPredictorFactory<DensePredictionMatrix<float64>> IScorePredictorFactory;
