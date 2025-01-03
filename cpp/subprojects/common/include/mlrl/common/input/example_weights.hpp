/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/util/dll_exports.hpp"

/**
 * Defines an interface for all classes that provide access to the weights of individual training examples.
 */
class MLRLCOMMON_API IExampleWeights {
    public:

      virtual ~IExampleWeights() {}
};
