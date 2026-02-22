/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_vector_nominal.hpp"

/**
 * A feature vector that stores the indices of the examples that are associated with each value, except for the majority
 * value, i.e., the most frequent value, of an ordinal feature.
 */
using OrdinalFeatureVector = NominalFeatureVector;
