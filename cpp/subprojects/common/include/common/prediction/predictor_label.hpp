/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/prediction_matrix_dense.hpp"
#include "common/output/prediction_matrix_sparse_binary.hpp"
#include "common/prediction/predictor.hpp"


/**
 * Defines an interface for all classes that allow to predict labels for given query examples.
 */
typedef IPredictor<DensePredictionMatrix<uint8>> ILabelPredictor;

/**
 * Defines an interface for all classes that allow to predict sparse labels for given query examples.
 */
typedef IPredictor<BinarySparsePredictionMatrix> ISparseLabelPredictor;
