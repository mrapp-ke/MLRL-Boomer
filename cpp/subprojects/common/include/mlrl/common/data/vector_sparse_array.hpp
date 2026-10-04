/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/indexed_value.hpp"
#include "mlrl/common/data/vector_dense.hpp"

/**
 * A vector that provides random read and write access, as well as read and write access via iterators, to elements,
 * consisting of an index and a corresponding value, stored in a newly allocated array.
 *
 * @tparam T                The type of the data that is stored in the vector
 * @tparam MemoryAllocator  The type of the memory allocator to be used
 */
template<typename T, typename MemoryAllocator = DefaultMemoryAllocator>
using SparseArrayVector = DenseVector<IndexedValue<T>, MemoryAllocator>;

/**
 * A vector that provides random read and write access, as well as read and write access via iterators, to elements,
 * consisting of an index and a corresponding value, stored in a newly allocated array, which can be resized.
 *
 * @tparam T                The type of the data that is stored in the vector
 * @tparam MemoryAllocator  The type of the memory allocator to be used
 */
template<typename T, typename MemoryAllocator = DefaultMemoryAllocator>
using ResizableSparseArrayVector = ResizableDenseVector<IndexedValue<T>, MemoryAllocator>;
