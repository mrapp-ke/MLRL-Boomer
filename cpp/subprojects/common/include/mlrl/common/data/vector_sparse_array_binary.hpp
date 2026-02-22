/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_dense.hpp"

/**
 * A vector that provides random read and write access, as well as read and write access via iterators, to indices
 * stored in a newly allocated array.
 */
using BinarySparseArrayVector = DenseVector<uint32>;

/**
 * A vector that provides random read and write access, as well as read and write access via iterators, to indices
 * stored in a newly allocated array, which can be resized
 */
using ResizableBinarySparseArrayVector = ResizableDenseVector<uint32>;
