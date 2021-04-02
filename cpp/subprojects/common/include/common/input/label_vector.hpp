/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/data/vector_dense.hpp"


/**
 * An one-dimensional sparse vector that stores the indices of labels that are relevant to an example.
 */
typedef DenseVector<uint32> LabelVector;
