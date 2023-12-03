/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_matrix_sparse_binary.hpp"

/**
 * Provides row-wise read and write access via iterators to the binary values stored in a sparse matrix.
 *
 * @tparam Matrix The type of the matrix
 */
template<typename Matrix>
using BinarySparseMatrixDecorator = IterableBinarySparseMatrixDecorator<MatrixDecorator<Matrix>>;
