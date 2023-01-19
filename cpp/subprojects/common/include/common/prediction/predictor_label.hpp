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
 * Defines an interface for all classes that allow to configure a `ILabelPredictor`.
 */
typedef IPredictorConfig<DensePredictionMatrix<uint8>> ILabelPredictorConfig;

/**
 * Defines an interface for all classes that allow to predict sparse labels for given query examples.
 */
typedef IPredictor<BinarySparsePredictionMatrix> ISparseLabelPredictor;

/**
 * Defines an interface for all classes that allow to create instances of the type `ISparseLabelPredictor`.
 */
typedef IPredictorFactory<BinarySparsePredictionMatrix> ISparseLabelPredictorFactory;

/**
 * Defines an interface for all classes that allow to configure a `ISparseLabelPredictor`.
 */
typedef IPredictorConfig<BinarySparsePredictionMatrix> ISparseLabelPredictorConfig;
