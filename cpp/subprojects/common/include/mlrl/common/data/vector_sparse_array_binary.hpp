/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_dense.hpp"

/**
 * A vector that provides random read and write access, as well as read and write access via iterators, to indices
 * stored in a newly allocated array.
 */
typedef DenseVector<uint32> BinarySparseArrayVector;

/**
 * A vector that provides random read and write access, as well as read and write access via iterators, to indices
 * stored in a newly allocated array, which can be resized
 */
typedef ResizableDenseVector<uint32> ResizableBinarySparseArrayVector;
