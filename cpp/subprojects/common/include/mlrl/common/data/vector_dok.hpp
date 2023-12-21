/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view_vector_dok.hpp"

/**
 * Provides read and write access via iterators to all non-zero values stored in vector in the dictionary of keys (DOK)
 * format.
 *
 * @tparam Vector The type of the vector
 */
template<typename Vector>
using DokVectorDecorator = IterableDokVectorDecorator<ViewDecorator<Vector>>;
