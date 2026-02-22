/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/vector_dok_binary.hpp"

/**
 * A view that provides access to the indices of training examples with missing values for a certain feature.
 */
using MissingFeatureVector = BinaryDokVector;

/**
 * Allocates the memory, a `MissingFeatureVector` provides access to.
 */
using AllocatedMissingFeatureVector = AllocatedBinaryDokVector;
