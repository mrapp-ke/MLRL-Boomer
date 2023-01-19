/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/prediction_matrix_dense.hpp"
#include "common/prediction/predictor.hpp"


/**
 * Defines an interface for all classes that allow to predict regression scores for given query examples.
 */
typedef IPredictor<DensePredictionMatrix<float64>> IScorePredictor;
