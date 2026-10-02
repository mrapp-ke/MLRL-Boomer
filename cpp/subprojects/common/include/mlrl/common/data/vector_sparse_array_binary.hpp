/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_dense.hpp"

/**
 * A vector that provides random read and write access, as well as read and write access via iterators, to indices
 * stored in a newly allocated array.
 *
 * @tparam MemoryAllocator The type of the memory allocator to be used
 */
template<typename MemoryAllocator = DefaultMemoryAllocator>
using BinarySparseArrayVector = DenseVector<uint32, MemoryAllocator>;

/**
 * A vector that provides random read and write access, as well as read and write access via iterators, to indices
 * stored in a newly allocated array, which can be resized
 */
using ResizableBinarySparseArrayVector = ResizableDenseVector<uint32>;
