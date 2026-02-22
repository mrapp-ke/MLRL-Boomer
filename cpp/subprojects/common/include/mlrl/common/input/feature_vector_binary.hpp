/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_vector_nominal.hpp"

/**
 * A feature vector that stores the indices of all examples that are associated with the minority value, i.e., the least
 * frequent value, of a binary feature.
 */
using BinaryFeatureVector = NominalFeatureVector;
